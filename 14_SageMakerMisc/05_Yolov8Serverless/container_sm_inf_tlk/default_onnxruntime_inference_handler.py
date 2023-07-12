# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import numpy as np
import onnxruntime as ort
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
    utils,
)

DEFAULT_MODEL_FILENAME = "model.onnx"


class ModelLoadError(Exception):
    pass


class DefaultONNXRuntimeInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)

    @staticmethod
    def _is_model_file(filename):
        is_model_file = False
        if os.path.isfile(filename):
            _, ext = os.path.splitext(filename)
            is_model_file = ext in [".onnx"]
        return is_model_file

    def default_model_fn(self, model_dir):
        """Loads a model. For ONNX Runtime, a default function to load a model only if Elastic Inference is used.
        In other cases, users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.

        Returns: An ONNX Runtime session.
        """
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            model_files = [file for file in os.listdir(model_dir) if self._is_model_file(file)]
            if len(model_files) != 1:
                raise ValueError(
                    "Exactly one .onnx file is required for PyTorch models: {}".format(model_files)
                )
            model_path = os.path.join(model_dir, model_files[0])
        try:
            model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using onnx export.".format(model_path)
            ) from e
        
        return model

    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized into numpy,
            depending if cuda is available.
        """
        np_array = decoder.decode(input_data, content_type)
        return np_array.astype(np.float32)

    def default_predict_fn(self, data, model):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn

        Returns: a prediction
        """
        input_name = model.get_inputs()[0].name # only 1 input is supported
        output = model.run(None, {input_name: data})

        return output

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        prediction = [p.tolist() for p in prediction]

        for content_type in utils.parse_accept(accept):
            if content_type in encoder.SUPPORTED_CONTENT_TYPES:
                encoded_prediction = encoder.encode(prediction, content_type)
                if content_type == content_types.CSV:
                    encoded_prediction = encoded_prediction.encode("utf-8")
                return encoded_prediction

        raise errors.UnsupportedFormatError(accept)
