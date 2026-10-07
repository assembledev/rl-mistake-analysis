"""Generate ONNX operator fixtures for the Rust inference tests."""

import struct
from pathlib import Path

import onnx
from onnx import TensorProto, helper

DIRECTORY = Path(__file__).resolve().parents[1] / "crates/inference/tests/fixtures"


def make_model(name, width=8, dtype=TensorProto.FLOAT, output_width=1, external=False):
    indices = helper.make_tensor(
        "indices", TensorProto.INT64, [1], struct.pack("<q", 0), raw=True
    )
    nodes = [
        helper.make_node("Gather", ["features", "indices"], ["selected"], axis=1),
        helper.make_node("Mul", ["selected", "selected"], ["squared"]),
    ]
    if output_width == 2:
        nodes.append(
            helper.make_node(
                "Concat", ["squared", "squared"], ["probabilities"], axis=1
            )
        )
    else:
        nodes.append(helper.make_node("Identity", ["squared"], ["probabilities"]))
    graph = helper.make_graph(
        nodes,
        "feature_square",
        [helper.make_tensor_value_info("features", dtype, ["N", width])],
        [helper.make_tensor_value_info("probabilities", dtype, ["N", output_width])],
        [indices],
    )
    model = helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model)
    path = DIRECTORY / name
    if external:
        weights = DIRECTORY / "external.data"
        weights.unlink(missing_ok=True)
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="external.data",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, path)


if __name__ == "__main__":
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    make_model("square.onnx")
    make_model("wrong_width.onnx", width=7)
    make_model("wrong_dtype.onnx", dtype=TensorProto.DOUBLE)
    make_model("wrong_output.onnx", output_width=2)
    make_model("external.onnx", external=True)
