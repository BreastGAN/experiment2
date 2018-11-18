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

import os
import numpy as np
from PIL import Image
import tensorflow as tf

from resources.data.features import numpy_to_feature, str_to_feature, int_to_feature
from resources.data.features import example_to_int, example_to_str, example_to_numpy

import skimage as ski

OPTIONS = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)


def get_image(example, feature_name, suffix=''):
    w = example_to_int(example, 'width' + suffix)
    h = example_to_int(example, 'height' + suffix)
    return example_to_numpy(example, feature_name + suffix, np.float32, (h, w))


def get_examples(tfrecords_glob, options=OPTIONS):
    for file in tf.gfile.Glob(tfrecords_glob):
        for record in tf.python_io.tf_record_iterator(file, options=options):
            example = tf.train.Example()
            example.ParseFromString(record)
            yield example


def to_png(matrix, path):
    im = Image.fromarray(matrix)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(path)


def feature_dict(img_path, old_path, bboxes, width, height, label, suffix=""):
    assert isinstance(label, np.int64)
    feature = {
            'path' + suffix: str_to_feature(old_path),
            'image_path' + suffix: str_to_feature(img_path),
            'bboxes' + suffix: numpy_to_feature(bboxes, np.float32),
            'width' + suffix: int_to_feature(width),
            'height' + suffix: int_to_feature(height),
            'label' + suffix: int_to_feature(label)
    }
    return feature


def to_example(feature_dict):
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def mask_to_boxes(mask_image):
    thresh = ski.filters.threshold_otsu(mask_image)
    bw = ski.morphology.closing(mask_image > thresh, ski.morphology.square(3))

    # Remove artifacts connected to image border
    cleared = ski.segmentation.clear_border(bw)

    lbl = ski.measure.label(cleared)
    props = ski.measure.regionprops(lbl)
    bboxes = []
    width = mask_image.shape[1]
    height = mask_image.shape[0]
    for prop in props:
        if prop.area > 10:
            y1, x1, y2, x2 = prop.bbox
            x1 = np.clip(float(x1), 0, width)
            x2 = np.clip(float(x2), 0, width)
            y1 = np.clip(float(y1), 0, height)
            y2 = np.clip(float(y2), 0, height)
            bboxes.append([x1, y1, x2, y2])
    return np.asarray(bboxes, dtype='float32')


def convert(examples, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    for example in examples:
        # Save image as png
        image = get_image(example, 'image')
        old_path = example_to_str(example, 'path')
        new_path = os.path.join(img_dir, old_path.replace('/', '_'))
        to_png(image, new_path)
        # Convert mask to bboxes
        mask = get_image(example, 'mask')
        bboxes = mask_to_boxes(mask)
        # Save it all into Example proto
        features = feature_dict(
                img_path=new_path,
                old_path=old_path,
                bboxes=bboxes,
                width=np.int64(image.shape[1]),
                height=np.int64(image.shape[0]),
                label=np.int64(example_to_int(example, 'label')))
        yield to_example(features)


def save_examples(examples, fname):
    with tf.python_io.TFRecordWriter(fname, options=OPTIONS) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_pattern", help="Global pattern for input tfrecords.")
    parser.add_argument("--output_file", help="Output file path.")
    args = parser.parse_args()

    tfrecords_glob = args.input_file_pattern
    output_file_path = args.output_file
    dir = os.path.dirname(output_file_path)
    img_dir = os.path.join(dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    input_examples = get_examples(tfrecords_glob)
    output_examples = convert(input_examples, img_dir)
    save_examples(output_examples, output_file_path)
