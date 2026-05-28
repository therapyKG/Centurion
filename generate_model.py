#!/usr/bin/env python3
"""
Generate an updatable MNIST CNN classifier for on-device training.

Architecture (matches Apple's updatable drawing classifier pattern):
  Input:  "image" — grayscale 28x28 image
  Output: "label" — digit 0-9

Conv(1->32, 3x3) -> ReLU -> Conv(32->64, 3x3) -> ReLU -> Flatten -> FC(64*24*24 -> 128) -> ReLU -> FC(128 -> 10) -> Softmax

Only the FC layers are marked updatable (convolutions are frozen feature extractors).
"""

import numpy as np
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes, MLModel

# --- 1. Build the CNN classifier ---

input_features = [("image", datatypes.Array(1, 28, 28))]  # CHW grayscale
output_features = [("label", datatypes.String())]

builder = NeuralNetworkBuilder(
    input_features=input_features,
    output_features=output_features,
    mode="classifier",
)

# Conv1: 1 -> 32 channels, 3x3, valid padding (output: 32x26x26)
np.random.seed(42)
W_conv1 = (np.random.randn(32, 1, 3, 3) * np.sqrt(2.0 / 9)).astype(np.float32)
b_conv1 = np.zeros(32, dtype=np.float32)
builder.add_convolution(
    name="conv1",
    kernel_channels=1,
    output_channels=32,
    height=3,
    width=3,
    stride_height=1,
    stride_width=1,
    border_mode="valid",
    groups=1,
    W=W_conv1,
    b=b_conv1,
    has_bias=True,
    input_name="image",
    output_name="conv1_out",
)

builder.add_activation(
    name="relu1",
    non_linearity="RELU",
    input_name="conv1_out",
    output_name="relu1_out",
)

# Conv2: 32 -> 64 channels, 3x3, valid padding (output: 64x24x24)
W_conv2 = (np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / (32 * 9))).astype(np.float32)
b_conv2 = np.zeros(64, dtype=np.float32)
builder.add_convolution(
    name="conv2",
    kernel_channels=32,
    output_channels=64,
    height=3,
    width=3,
    stride_height=1,
    stride_width=1,
    border_mode="valid",
    groups=1,
    W=W_conv2,
    b=b_conv2,
    has_bias=True,
    input_name="relu1_out",
    output_name="conv2_out",
)

builder.add_activation(
    name="relu2",
    non_linearity="RELU",
    input_name="conv2_out",
    output_name="relu2_out",
)

# Flatten: 64 * 24 * 24 = 36864
builder.add_flatten(
    name="flatten",
    mode=0,  # channel-first
    input_name="relu2_out",
    output_name="flatten_out",
)

# FC1: 36864 -> 128
flat_size = 64 * 24 * 24
W_fc1 = (np.random.randn(128, flat_size) * np.sqrt(2.0 / flat_size)).astype(np.float32)
b_fc1 = np.zeros(128, dtype=np.float32)
builder.add_inner_product(
    name="fc1",
    W=W_fc1,
    b=b_fc1,
    input_channels=flat_size,
    output_channels=128,
    has_bias=True,
    input_name="flatten_out",
    output_name="fc1_out",
)

builder.add_activation(
    name="relu3",
    non_linearity="RELU",
    input_name="fc1_out",
    output_name="relu3_out",
)

# FC2: 128 -> 10 (digits 0-9)
W_fc2 = (np.random.randn(10, 128) * np.sqrt(2.0 / 128)).astype(np.float32)
b_fc2 = np.zeros(10, dtype=np.float32)
builder.add_inner_product(
    name="fc2",
    W=W_fc2,
    b=b_fc2,
    input_channels=128,
    output_channels=10,
    has_bias=True,
    input_name="relu3_out",
    output_name="fc2_out",
)

# Softmax
builder.add_softmax(
    name="softmax",
    input_name="fc2_out",
    output_name="classProbabilities",
)

# --- 2. Set class labels ---

spec = builder.spec
class_labels = spec.neuralNetworkClassifier.stringClassLabels
for i in range(10):
    class_labels.vector.append(str(i))

spec.description.predictedFeatureName = "label"
spec.description.predictedProbabilitiesName = "classProbabilities"

# Add classProbabilities as an output
prob_output = spec.description.output.add()
prob_output.name = "classProbabilities"
prob_output.type.dictionaryType.stringKeyType.MergeFrom(
    ct.proto.FeatureTypes_pb2.StringFeatureType()
)

# Fix the input type to be an image (grayscale 28x28)
input_desc = spec.description.input[0]
input_desc.type.ClearField("multiArrayType")
input_desc.type.imageType.width = 28
input_desc.type.imageType.height = 28
input_desc.type.imageType.colorSpace = ct.proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE

# --- 3. Mark as updatable (only FC layers — conv layers are frozen) ---

builder.make_updatable(["fc1", "fc2"])

builder.set_categorical_cross_entropy_loss(
    name="lossLayer",
    input="classProbabilities",
)

from coremltools.models.neural_network import SgdParams
builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=32, momentum=0.9))

# Epochs: range 1..100
epochs_param = spec.neuralNetworkClassifier.updateParams.epochs
epochs_param.defaultValue = 5
epochs_param.range.minValue = 1
epochs_param.range.maxValue = 100

# Spec version must be >= 4 for updatable models
spec.specificationVersion = max(spec.specificationVersion, 4)

# --- 4. Save ---

output_path = "Centurion/MNISTClassifier.mlmodel"
ct.utils.save_spec(spec, output_path)
print(f"Saved updatable model to: {output_path}")

# Verify
loaded = MLModel(output_path)
s = loaded.get_spec()
print(f"Spec version: {s.specificationVersion}")
print(f"Is updatable: {s.isUpdatable}")
print(f"Inputs: {[(i.name, str(i.type.WhichOneof('Type'))) for i in s.description.input]}")
print(f"Outputs: {[(o.name, str(o.type.WhichOneof('Type'))) for o in s.description.output]}")
print(f"Training inputs: {[(t.name, str(t.type.WhichOneof('Type'))) for t in s.description.trainingInput]}")
updatable_layers = [l.name for l in s.neuralNetworkClassifier.layers if l.isUpdatable]
all_layers = [l.name for l in s.neuralNetworkClassifier.layers]
print(f"All layers: {all_layers}")
print(f"Updatable layers: {updatable_layers}")

import os
size_kb = os.path.getsize(output_path) / 1024
print(f"Model size: {size_kb:.0f} KB")
