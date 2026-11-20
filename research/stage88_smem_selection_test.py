from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/cubek-matmul-0.3.1/src/multi_level/routines/selector/plane.rs"


def exact_function(text: str, name: str) -> str:
    marker = f"fn {name}("
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f"Function not found in vendor implementation: {name}")
    begin = text.find("{", pos)
    if begin < 0:
        raise RuntimeError(f"Function body not found: {name}")
    depth = 0
    for end in range(begin, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[pos:end + 1]
    raise RuntimeError(f"Unclosed function body: {name}")


TESTS = r'''
#[derive(Clone, Copy)]
enum AccumulatorOperand { Absent, Present }
#[derive(Debug)]
enum MatmulAvailabilityError { SharedMemoryTooBig {requested:usize, available:usize} }
#[derive(Clone, Copy)]
struct Elem(usize);
impl Elem { fn size(&self) -> usize { self.0 } }
struct MatmulElems {lhs_stage:Elem, rhs_stage:Elem, acc_stage:Elem}
fn dtype(size:usize, out:usize) -> MatmulElems {
  MatmulElems {lhs_stage:Elem(size),rhs_stage:Elem(size),acc_stage:Elem(out)}
}
#[test]
fn fit_4060_with_bias() {
 let d = dtype(2,4);
 let stage = (1,16,16,2);
 let tile = (16,16,16);
 assert!(plane_tiling_smem_bytes(stage,tile,&d,1,AccumulatorOperand::Present)>101376);
 let result = fit_plane_tiling_to_smem(stage,tile,&d,1,AccumulatorOperand::Present,101376).unwrap();
 assert!(plane_tiling_smem_bytes(result,tile,&d,1,AccumulatorOperand::Present)<=101376);
}
#[test]
fn bias_accounted_for() {
 let d=dtype(2,4);let stage=(2,4,4,2);let tile=(16,16,16);
 assert!(plane_tiling_smem_bytes(stage,tile,&d,1,AccumulatorOperand::Present)>
  plane_tiling_smem_bytes(stage,tile,&d,1,AccumulatorOperand::Absent));
}
#[test]
fn unchanged_when_fits() {
 let d=dtype(4,4);let stage=(1,2,2,2);
 assert_eq!(fit_plane_tiling_to_smem(stage,(16,16,16),&d,1,AccumulatorOperand::Absent,101376).unwrap(),stage);
}
#[test]
fn no_invalid_undersized_configuration() {
 let d=dtype(2,4);
 assert!(matches!(fit_plane_tiling_to_smem((8,8,8,4),(16,16,16),&d,1,AccumulatorOperand::Present,1),
 Err(MatmulAvailabilityError::SharedMemoryTooBig{..})));
}
#[test]
fn hardware_and_precision_sweep() {
 for elt in [2,4] {for acc in [2,4] {for buffered in [1,2] {
 for limit in [24576usize,32768,49152,65536,101376,163840] {
  let d=dtype(elt,acc);
  let x=fit_plane_tiling_to_smem((2,16,16,4),(16,16,16),&d,buffered,AccumulatorOperand::Present,limit).unwrap();
  assert!(plane_tiling_smem_bytes(x,(16,16,16),&d,buffered,AccumulatorOperand::Present)<=limit);
 }}}}
}
'''


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    functions = "\n\n".join(exact_function(source, name) for name in (
        "plane_tiling_smem_bytes", "fit_plane_tiling_to_smem"))
    with tempfile.TemporaryDirectory(prefix="stage88-smem-") as directory:
        folder = Path(directory)
        file = folder / "test.rs"
        binary = folder / ("test.exe" if os.name == "nt" else "test")
        file.write_text(TESTS + "\n" + functions + "\n", encoding="utf-8")
        subprocess.run(["rustc", "--edition=2024", "--test", str(file), "-o", str(binary)], check=True)
        subprocess.run([str(binary), "--nocapture"], check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Stage88 isolated source-function test failed: {exc}", file=sys.stderr)
        raise
