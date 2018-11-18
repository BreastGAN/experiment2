# Copyright 2018 Lukas Jendele and Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from models.breast_cycle_gan.data_provider import parse_example

COMPRESSION_TYPE = tf.python_io.TFRecordCompressionType.GZIP
OPTIONS = tf.python_io.TFRecordOptions(compression_type=COMPRESSION_TYPE)


def int_to_feature(i):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))


def str_to_feature(s):
    str_bytes = tf.compat.as_bytes(s)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str_bytes]))


def example_to_str(example, feature_name):
    return tf.compat.as_text(example.features.feature[feature_name].bytes_list.value[0])


def example_to_int(example, feature_name):
    return example.features.feature[feature_name].int64_list.value[0]


def example_to_numpy(example, feature_name, dtype, shape):
    arr_string = example.features.feature[feature_name].bytes_list.value[0]
    arr_1d = np.frombuffer(arr_string, dtype=dtype)
    return np.reshape(arr_1d, shape)


def numpy_to_feature(arr, dtype):
    return str_to_feature(arr.astype(dtype).tostring())


def img_to_feature(img):
    return str_to_feature(img.astype(np.float32).tostring())


def to_feature_dict(img_path, img, mask, width, height, label, suffix=""):
    assert img.shape == mask.shape
    assert isinstance(label, np.int64)
    assert isinstance(img_path, str)
    feature = {
            'path' + suffix: str_to_feature(img_path),
            'image' + suffix: img_to_feature(img),
            'mask' + suffix: img_to_feature(mask),
            'width': int_to_feature(width),
            'height': int_to_feature(height),
            'label' + suffix: int_to_feature(label)
    }
    return feature


def to_example(feature_dict):
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def show_records(path, img_size=(256, 256)):
    # Lazy import
    import matplotlib.pyplot as plt
    i = 0
    for record in tf.python_io.tf_record_iterator(path, options=OPTIONS):
        print(str(i))
        concat, img, mask = parse_example(record, img_size=img_size)
        plt.imshow(img)
        plt.show()
        plt.imshow(mask)
        plt.show()
        i += 1
        if i == 3:
            break
