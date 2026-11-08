#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

"""Generate a model whose only invalid value is seen after identity elimination.

Constant -> Identity -> Upsample: the scales are Dynamic while types are inferred and only
become a constant once the Identity is removed, so Upsample first validates them when the
node is built. Scales of 1.75 on a 3x3 input do not divide it evenly, which Upsample rejects.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def main():
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 1.75, 1.75], dtype=np.float32))
    nodes = [
        helper.make_node("Constant", [], ["scales_const"], value=scales),
        helper.make_node("Identity", ["scales_const"], ["scales"]),
        helper.make_node("Upsample", ["x", "scales"], ["y"], mode="nearest"),
    ]
    graph = helper.make_graph(
        nodes,
        "late_lifted_constant",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5, 5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    path = os.path.join(FIXTURES_DIR, "late_lifted_constant.onnx")
    onnx.save(model, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
