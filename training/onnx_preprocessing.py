"""ONNX conversion for median imputation with missing-value indicators."""

import numpy as np
from onnx import TensorProto
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from sklearn.impute import SimpleImputer


def imputer_shape(operator):
    model = operator.raw_operator
    if (
        model.strategy != "median"
        or not model.keep_empty_features
        or not model.add_indicator
    ):
        raise ValueError(
            "Export requires median imputation with preserved columns and indicators"
        )
    operator.outputs[0].type = FloatTensorType(
        [
            operator.inputs[0].type.shape[0],
            len(model.statistics_) + len(model.indicator_.features_),
        ]
    )


def convert_imputer(scope, operator, container):
    model = operator.raw_operator

    def constant(name, values, data_type):
        name = scope.get_unique_variable_name(name)
        container.add_initializer(
            name, data_type, values.shape, values.flatten().tolist()
        )
        return name

    def node(operation, inputs, **attributes):
        output = scope.get_unique_variable_name(operation.lower())
        container.add_node(operation, inputs, [output], **attributes)
        return output

    features = operator.inputs[0].full_name
    missing = node("IsNaN", [features])
    medians = constant(
        "medians", np.asarray(model.statistics_, dtype=np.float32), TensorProto.FLOAT
    )
    prepared = node("Where", [missing, medians, features])
    if len(model.indicator_.features_):
        columns = constant(
            "indicator_columns",
            model.indicator_.features_.astype(np.int64),
            TensorProto.INT64,
        )
        flags = node("Gather", [missing, columns], axis=1)
        flags = node("Cast", [flags], to=TensorProto.FLOAT)
        prepared = node("Concat", [prepared, flags], axis=1)
    container.add_node("Identity", [prepared], [operator.outputs[0].full_name])


def register_imputer():
    update_registered_converter(
        SimpleImputer, "MedianImputerWithIndicators", imputer_shape, convert_imputer
    )
