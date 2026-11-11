//! Opt-in resident CubeCL IndexIVFFlat search. No query or retrieval readback.
//! Supports exact IVF nprobe=1 with eight ordered neighbors; the reference CPU
//! path remains authoritative until full CUDA/WGPU benchmark and parity review.

use crate::faiss_index::{IvfFlatIndex, PackedGpuIndex};
use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::cubecl::{bytes::Bytes, client::Client, server::Handle};
use burn::tensor::{DType, Device, Tensor};
use burn_cubecl::tensor::CubeTensor;

#[cfg(feature = "cuda")]
type CubeBackend = burn::backend::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
type CubeBackend = burn::backend::Cpu;
#[cfg(all(not(feature = "cuda"), not(feature = "cpu"), feature = "wgpu"))]
type CubeBackend = burn::backend::Wgpu;

const K: usize = 8;
const WORKGROUP: u32 = 64;

fn byte_vec_f32(values: Vec<f32>) -> Bytes {
    Bytes::from_bytes_vec(values.into_iter().flat_map(f32::to_ne_bytes).collect())
}
fn byte_vec_u32(values: Vec<u32>) -> Bytes {
    Bytes::from_bytes_vec(values.into_iter().flat_map(u32::to_ne_bytes).collect())
}

/// One work item for each frame/centroid pair. A frame is never transferred
/// to the host merely to select the nearest IVF list.
#[cube(launch_unchecked)]
fn centroid_dist_kernel(
    queries: &[f32],
    centroids: &[f32],
    distances: &mut [f32],
    rows: usize,
    frames: usize,
    nlist: usize,
    dim: usize,
    stride_batch: usize,
    stride_frame: usize,
    stride_channel: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < rows * nlist {
        let row = idx / nlist;
        let centroid = idx % nlist;
        let q = (row / frames) * stride_batch + (row % frames) * stride_frame;
        let c = centroid * dim;
        let mut distance = 0.0_f32;
        for d in 0..dim {
            let delta = queries[q + d * stride_channel] - centroids[c + d];
            distance += delta * delta;
        }
        distances[idx] = distance;
    }
}

#[cube(launch_unchecked)]
fn closest_centroid_kernel(distances: &[f32], selected: &mut [u32], rows: usize, nlist: usize) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let mut best = distances[row * nlist];
        let mut best_id = 0_u32;
        for i in 1..nlist {
            let value = distances[row * nlist + i];
            if value < best {
                best = value;
                best_id = i as u32;
            }
        }
        selected[row] = best_id;
    }
}

/// Candidate distances are parallel across the selected inverted list.
#[cube(launch_unchecked)]
fn candidate_dist_kernel(
    queries: &[f32],
    codes: &[f32],
    offsets: &[u32],
    selected: &[u32],
    distances: &mut [f32],
    rows: usize,
    frames: usize,
    max_list: usize,
    dim: usize,
    stride_batch: usize,
    stride_frame: usize,
    stride_channel: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < rows * max_list {
        let row = idx / max_list;
        let slot = idx % max_list;
        let list = selected[row] as usize;
        let begin = offsets[list] as usize;
        let end = offsets[list + 1] as usize;
        if begin + slot < end {
            let q = (row / frames) * stride_batch + (row % frames) * stride_frame;
            let v = (begin + slot) * dim;
            let mut distance = 0.0_f32;
            for d in 0..dim {
                let delta = queries[q + d * stride_channel] - codes[v + d];
                distance += delta * delta;
            }
            distances[idx] = distance;
        } else {
            distances[idx] = f32::MAX;
        }
    }
}

/// Deterministic top-8 by squared distance, then original vector ID.
/// A list containing fewer than eight vectors returns unchanged features.
#[cube(launch_unchecked)]
fn rank_kernel(
    distances: &[f32],
    ids: &[u32],
    offsets: &[u32],
    selected: &[u32],
    picked: &mut [u32],
    weights: &mut [f32],
    rows: usize,
    max_list: usize,
) {
    let row = ABSOLUTE_POS as usize;
    if row < rows {
        let list = selected[row] as usize;
        let begin = offsets[list] as usize;
        let end = offsets[list + 1] as usize;
        let length = end - begin;
        let target = row * 8;
        if length < 8 {
            picked[target] = u32::MAX;
        } else {
            for rank in 0..8 {
                let mut best = distances[row * max_list];
                let mut best_id = ids[begin];
                let mut best_slot = selected[row];
                let mut found = false;
                for slot in 0..length {
                    let candidate = distances[row * max_list + slot];
                    let id = ids[begin + slot];
                    let mut used = false;
                    for prev in 0..rank {
                        if picked[target + prev] == (begin + slot) as u32 {
                            used = true;
                        }
                    }
                    if !used && (!found || candidate < best || (candidate == best && id < best_id))
                    {
                        best = candidate;
                        best_id = id;
                        best_slot = slot as u32;
                        found = true;
                    }
                }
                picked[target + rank] = (begin + best_slot as usize) as u32;
            }
            let mut zeros = 0_u32;
            let mut sum = 0.0_f32;
            for rank in 0..8 {
                let loc = picked[target + rank] as usize - begin;
                let dist = distances[row * max_list + loc];
                if dist == 0.0_f32 {
                    zeros += 1;
                } else {
                    let safe = if dist < f32::MIN_POSITIVE {
                        f32::MIN_POSITIVE
                    } else {
                        dist
                    };
                    let raw = 1.0_f32 / (safe * safe);
                    sum += raw;
                }
            }
            for rank in 0..8 {
                let loc = picked[target + rank] as usize - begin;
                let dist = distances[row * max_list + loc];
                if zeros > 0 {
                    weights[target + rank] = if dist == 0.0_f32 {
                        1.0_f32 / zeros as f32
                    } else {
                        0.0_f32
                    };
                } else {
                    let safe = if dist < f32::MIN_POSITIVE {
                        f32::MIN_POSITIVE
                    } else {
                        dist
                    };
                    weights[target + rank] = if !(sum > 0.0_f32 && sum <= f32::MAX) {
                        f32::NAN
                    } else {
                        (1.0_f32 / (safe * safe)) / sum
                    };
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
fn blend_kernel(
    queries: &[f32],
    codes: &[f32],
    picked: &[u32],
    weights: &[f32],
    rates: &[f32],
    output: &mut [f32],
    rows: usize,
    frames: usize,
    dim: usize,
    stride_batch: usize,
    stride_frame: usize,
    stride_channel: usize,
) {
    let idx = ABSOLUTE_POS as usize;
    if idx < rows * dim {
        let row = idx / dim;
        let ch = idx % dim;
        let original = queries
            [(row / frames) * stride_batch + (row % frames) * stride_frame + ch * stride_channel];
        let rate = rates[row / frames];
        if rate == 0.0_f32 || picked[row * 8] == u32::MAX {
            output[idx] = original;
        } else {
            let mut result = 0.0_f32;
            for k in 0..8 {
                let position = picked[row * 8 + k] as usize;
                result += weights[row * 8 + k] * codes[position * dim + ch];
            }
            output[idx] = original * (1.0_f32 - rate) + result * rate;
        }
    }
}

pub(crate) struct GpuIvfIndex {
    client: Client,
    device: burn_cubecl::CubeDevice,
    centroids: Handle,
    codes: Handle,
    offsets: Handle,
    ids: Handle,
    nlist: usize,
    dim: usize,
    max_list: usize,
    total: usize,
}

impl GpuIvfIndex {
    pub(crate) fn new(index: &IvfFlatIndex, device: &Device) -> Result<Self, String> {
        Self::from_packed(index.packed_gpu_data()?, device)
    }

    fn from_packed(packed: PackedGpuIndex, device: &Device) -> Result<Self, String> {
        let PackedGpuIndex {
            dimension,
            nprobe,
            centroids,
            codes,
            ids,
            offsets,
            max_list,
        } = packed;
        if nprobe != 1 {
            return Err("GPU IVF currently supports exact nprobe=1 only".into());
        }
        if max_list < 8 {
            return Err("GPU IVF requires at least eight vectors in one IVF list".into());
        }
        let marker = Tensor::<1>::zeros([1], device)
            .try_into_primitive::<CubeBackend>()
            .map_err(|e| {
                format!("GPU IVF requires a direct CubeCL backend (no CPU fallback): {e:?}")
            })?;
        let client = marker.client;
        let device = marker.device;
        let nlist = offsets.len() - 1;
        let total = ids.len();
        if std::env::var_os("RVC_PROFILE").is_some() {
            eprintln!(
                "profile gpu_ivf_index resident_vectors={} dimensions={} centroids={} index_storage_bytes={} max_list={}",
                total,
                dimension,
                nlist,
                (centroids.len() + codes.len()) * 4 + (ids.len() + offsets.len()) * 4,
                max_list
            );
        }
        Ok(Self {
            centroids: client.create(byte_vec_f32(centroids)),
            codes: client.create(byte_vec_f32(codes)),
            ids: client.create(byte_vec_u32(ids)),
            offsets: client.create(byte_vec_u32(offsets)),
            client,
            device,
            nlist,
            dim: dimension,
            max_list,
            total,
        })
    }

    pub(crate) fn blend(&self, content: Tensor<3>, rates: &[f32]) -> Result<Tensor<3>, String> {
        let [batch, frames, dim] = content.dims();
        if batch == 0
            || frames == 0
            || dim != self.dim
            || rates.len() != batch
            || rates
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("GPU IVF feature dimensions or rates are invalid".into());
        }
        let query = content
            .try_into_primitive::<CubeBackend>()
            .map_err(|e| format!("GPU IVF received non-CubeCL features: {e:?}"))?;
        if query.dtype != DType::F32 || query.meta.is_tiled() || query.device != self.device {
            return Err(
                "GPU IVF requires FP32 features resident on its own non-tiled GPU device".into(),
            );
        }
        let strides = query.meta.strides();
        let required = (batch - 1)
            .checked_mul(strides[0])
            .and_then(|x| x.checked_add((frames - 1).checked_mul(strides[1])?))
            .and_then(|x| x.checked_add((dim - 1).checked_mul(strides[2])?))
            .and_then(|x| x.checked_add(1))
            .ok_or("GPU IVF feature stride overflow")?;
        if required
            .checked_mul(4)
            .ok_or("GPU IVF allocation overflow")?
            > query.handle.size_in_used() as usize
        {
            return Err("GPU IVF feature tensor exceeds its allocation".into());
        }
        let rows = batch.checked_mul(frames).ok_or("GPU IVF row overflow")?;
        let ndist = rows
            .checked_mul(self.nlist)
            .ok_or("GPU IVF centroid size overflow")?;
        let ncandidate = rows
            .checked_mul(self.max_list)
            .ok_or("GPU IVF list size overflow")?;
        let noutput = rows
            .checked_mul(dim)
            .ok_or("GPU IVF output size overflow")?;
        for elems in [rows, ndist, ncandidate, noutput] {
            if elems.div_ceil(WORKGROUP as usize) > u32::MAX as usize {
                return Err("GPU IVF launch grid exceeds CubeCL limits".into());
            }
        }
        let client = &query.client;
        let bytes = |elements: usize| {
            elements
                .checked_mul(4)
                .ok_or("GPU IVF allocation size overflow")
        };
        let rate_handle = client.create(byte_vec_f32(rates.to_vec()));
        let centroid_dist = client.empty(bytes(ndist)?);
        let chosen = client.empty(bytes(rows)?);
        let candidates = client.empty(bytes(ncandidate)?);
        let picked = client.empty(bytes(
            rows.checked_mul(K).ok_or("GPU IVF rank count overflow")?,
        )?);
        let weights = client.empty(bytes(
            rows.checked_mul(K)
                .ok_or("GPU IVF weights count overflow")?,
        )?);
        let output = client.empty(bytes(noutput)?);
        let arg = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len) };
        unsafe {
            centroid_dist_kernel::launch_unchecked(
                client,
                CubeCount::Static(ndist.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(WORKGROUP),
                arg(&query.handle, query.handle.size_in_used() as usize / 4),
                arg(&self.centroids, self.nlist * dim),
                arg(&centroid_dist, ndist),
                rows,
                frames,
                self.nlist,
                dim,
                strides[0],
                strides[1],
                strides[2],
            );
            closest_centroid_kernel::launch_unchecked(
                client,
                CubeCount::Static(rows.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(WORKGROUP),
                arg(&centroid_dist, ndist),
                arg(&chosen, rows),
                rows,
                self.nlist,
            );
            candidate_dist_kernel::launch_unchecked(
                client,
                CubeCount::Static(ncandidate.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(WORKGROUP),
                arg(&query.handle, query.handle.size_in_used() as usize / 4),
                arg(&self.codes, self.total * dim),
                arg(&self.offsets, self.nlist + 1),
                arg(&chosen, rows),
                arg(&candidates, ncandidate),
                rows,
                frames,
                self.max_list,
                dim,
                strides[0],
                strides[1],
                strides[2],
            );
            rank_kernel::launch_unchecked(
                client,
                CubeCount::Static(rows.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(WORKGROUP),
                arg(&candidates, ncandidate),
                arg(&self.ids, self.total),
                arg(&self.offsets, self.nlist + 1),
                arg(&chosen, rows),
                arg(&picked, rows * K),
                arg(&weights, rows * K),
                rows,
                self.max_list,
            );
            blend_kernel::launch_unchecked(
                client,
                CubeCount::Static(noutput.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(WORKGROUP),
                arg(&query.handle, query.handle.size_in_used() as usize / 4),
                arg(&self.codes, self.total * dim),
                arg(&picked, rows * K),
                arg(&weights, rows * K),
                arg(&rate_handle, batch),
                arg(&output, noutput),
                rows,
                frames,
                dim,
                strides[0],
                strides[1],
                strides[2],
            );
        }
        Ok(Tensor::from_primitive::<CubeBackend>(
            CubeTensor::new_contiguous(
                self.client.clone(),
                self.device.clone(),
                [batch, frames, dim].into(),
                output,
                DType::F32,
            ),
        ))
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn test_index() -> PackedGpuIndex {
        let dimension = 4;
        let mut centroids = Vec::new();
        let mut codes = Vec::new();
        let mut ids = Vec::new();
        let mut offsets = Vec::new();
        for list in 0..3_u32 {
            let position = list as f32 * 10.0;
            centroids.extend([position; 4]);
            offsets.push(ids.len() as u32);
            for slot in 0..9_u32 {
                let deviation = (slot as f32 - 4.0) * 0.125;
                codes.extend([
                    position + deviation,
                    position - deviation,
                    position,
                    position,
                ]);
                ids.push(list * 100 + (8 - slot));
            }
        }
        offsets.push(ids.len() as u32);
        PackedGpuIndex {
            dimension,
            nprobe: 1,
            centroids,
            codes,
            ids,
            offsets,
            max_list: 9,
        }
    }

    fn reference(index: &PackedGpuIndex, query: &[f32], rate: f32) -> Vec<f32> {
        if rate == 0.0 {
            return query.to_vec();
        }
        let centroid = index
            .centroids
            .chunks_exact(index.dimension)
            .enumerate()
            .map(|(i, c)| {
                let distance: f32 = query.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, distance)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .unwrap()
            .0;
        let start = index.offsets[centroid] as usize;
        let end = index.offsets[centroid + 1] as usize;
        let mut neighbors: Vec<_> = (start..end)
            .map(|i| {
                let data = &index.codes[i * index.dimension..(i + 1) * index.dimension];
                let distance: f32 = query.iter().zip(data).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, distance, index.ids[i])
            })
            .collect();
        neighbors.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)));
        if neighbors.len() < 8 {
            return query.to_vec();
        }
        neighbors.truncate(8);
        let zeros = neighbors.iter().filter(|x| x.1 == 0.0).count();
        let weights: Vec<f32> = if zeros != 0 {
            neighbors
                .iter()
                .map(|x| if x.1 == 0.0 { 1.0 / zeros as f32 } else { 0.0 })
                .collect()
        } else {
            let raw: Vec<f32> = neighbors.iter().map(|x| 1.0 / (x.1 * x.1)).collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|x| x / sum).collect()
        };
        (0..index.dimension)
            .map(|d| {
                let mut weighted = 0.0_f32;
                for (neighbor, weight) in neighbors.iter().zip(weights.iter()) {
                    weighted += *weight * index.codes[neighbor.0 * index.dimension + d];
                }
                query[d] * (1.0 - rate) + weighted * rate
            })
            .collect()
    }

    #[test]
    #[ignore = "opt-in private CrazyMita FAISS; full device-resident IVF parity"]
    fn real_ivf_gpu_search_matches_cpu_reference_within_tolerance() {
        let path = std::env::var("RVC_REAL_FAISS_INDEX").expect("RVC_REAL_FAISS_INDEX required");
        let index = IvfFlatIndex::open(path).unwrap();
        assert_eq!(index.nprobe(), 1);
        let device = Device::cpu();
        let gpu = GpuIvfIndex::new(&index, &device).unwrap();
        let batch = 2;
        let frames = 5;
        let channels = index.dimension();
        let rates = [0.75, 1.0];
        let mut input = Vec::new();
        for row in 0..batch * frames {
            let source = index.reconstruct((row * 7919 + 19) % index.len()).unwrap();
            for (channel, &value) in source.iter().enumerate() {
                input.push(value + (((row * 17 + channel * 13) % 11) as f32 - 5.0) * 0.0007);
            }
        }
        let output = gpu
            .blend(
                Tensor::<3>::from_data(
                    TensorData::new(input.clone(), [batch, frames, channels]),
                    &device,
                ),
                &rates,
            )
            .unwrap()
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        let mut cpu = input;
        for (b, &rate) in rates.iter().enumerate() {
            index
                .blend_features(
                    &mut cpu[b * frames * channels..(b + 1) * frames * channels],
                    rate,
                )
                .unwrap();
        }
        let max_abs = cpu
            .iter()
            .zip(&output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let mismatches = cpu
            .iter()
            .zip(&output)
            .filter(|(a, b)| (**a - **b).abs() > 1e-4)
            .count();
        eprintln!(
            "REAL GPU-IVF CPU/LLVM total={} max_abs={max_abs:.9} count_above_1e-4={mismatches}",
            output.len()
        );
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(
            max_abs <= 1e-4,
            "GPU-resident IVF differs from CPU reference: {max_abs}"
        );
    }

    #[test]
    fn ivf_cuda_style_kernels_run_on_cpu_and_preserve_batch_rates() {
        let fixture = test_index();
        let device = Device::cpu();
        let index = GpuIvfIndex::from_packed(test_index(), &device).unwrap();
        let rates = [0.75_f32, 0.5, 0.0];
        let mut queries = Vec::new();
        for batch in 0..3 {
            for frame in 0..3 {
                let pos = (batch * 3 + frame) as f32 * 3.7;
                queries.extend([pos, pos + 0.07, pos + 0.14, pos + 0.21]);
            }
        }
        let tensor = Tensor::<3>::from_data(TensorData::new(queries.clone(), [3, 3, 4]), &device);
        let result = index
            .blend(tensor, &rates)
            .unwrap()
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        for (i, query) in queries.chunks_exact(4).enumerate() {
            let expected = reference(&fixture, query, rates[i / 3]);
            for d in 0..4 {
                assert!(
                    (expected[d] - result[i * 4 + d]).abs() <= 2e-5,
                    "row={i} dim={d}: {} != {}",
                    expected[d],
                    result[i * 4 + d]
                );
            }
        }
    }
}
