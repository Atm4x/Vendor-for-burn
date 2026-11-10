//! Cross-runtime one-time packing for constant Conv1d inference weights.
//!
//! Convert [Cout, K, CinGroup] into [Groups, K, CinGroup, CoutGroup]
//! without a host readback. The result is consumed by `direct_prepacked`.

use cubecl::{calculate_cube_count_elemwise, client::Client, prelude::*};

use crate::components::ConvSetupError;

#[cube(launch_unchecked, address_type = "dynamic")]
fn pack_conv1d_weights_kernel<E: Numeric>(
    source: &Tensor<E>,
    output: &mut Tensor<E>,
    #[define(E)] _dtype: ElemType,
) {
    let pos = ABSOLUTE_POS;
    let cout_per_group = output.shape(3);
    let cin_per_group = output.shape(2);
    let k_size = output.shape(1);
    let total = output.shape(0) * k_size * cin_per_group * cout_per_group;
    if pos < total {
        let oc = pos % cout_per_group;
        let x = pos / cout_per_group;
        let ci = x % cin_per_group;
        let k = (x / cin_per_group) % k_size;
        let group = x / (cin_per_group * k_size);
        let source_oc = group * cout_per_group + oc;
        let src_pos = source_oc * source.stride(0) + k * source.stride(1) + ci;
        output[pos] = source[src_pos];
    }
}

/// Launch the same packing kernel on CubeCL CPU, CUDA, ROCm or WGPU.
/// The destination is owned by the model's inference plan and reused for every forward.
pub fn launch_pack_conv1d_weights(
    client: &Client,
    source: TensorBinding,
    target: TensorBinding,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    if source.shape.len() != 3
        || target.shape.len() != 4
        || target.shape.iter().any(|&v| v == 0)
        || source.shape[0] != target.shape[0] * target.shape[3]
        || source.shape[1] != target.shape[1]
        || source.shape[2] != target.shape[2]
    {
        return Err(ConvSetupError::Unknown);
    }
    let total = target.shape.iter().product::<usize>();
    let cube_dim = CubeDim::new(client, total);
    let cube_count = calculate_cube_count_elemwise(client, total, cube_dim);
    let address_type = source
        .required_address_type(dtype.size())
        .max(target.required_address_type(dtype.size()));
    unsafe {
        pack_conv1d_weights_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            source.into_tensor_arg(),
            target.into_tensor_arg(),
            dtype,
        );
    }
    Ok(())
}
