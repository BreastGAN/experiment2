# Copyright 2018 Lukas Jendele and Ondrej Skopek.
# Adapted from The TensorFlow Authors, under the ASL 2.0.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import models.breast_cycle_gan.data_provider as data_provider
import models.breast_cycle_gan.generator as generator
import resources.data.features as features
from resources.data.transformer import OPTIONS

flags = tf.flags
tfgan = tf.contrib.gan

flags.DEFINE_bool("use_icnr", False, "Use kernel initialization as described in the Twitter paper.")

flags.DEFINE_string('upsample_method', 'conv2d_transpose', 'Upsampling method for genrator.')

flags.DEFINE_string('checkpoint_path', '', 'CycleGAN checkpoint path created by train.py. '
                    '(e.g. "/mylogdir/model.ckpt-18442")')

flags.DEFINE_string('image_source', '/scratch_net/biwidl100/oskopek/cbis/healthy.tfrecord',
                    'File pattern of images in image set X')

flags.DEFINE_string('generated_dir', '/tmp/generated/', 'Where to output the generated images.')

flags.DEFINE_integer('height', 512, 'Image height')

flags.DEFINE_integer('width', 408, 'Image width')

flags.DEFINE_enum('model', 'H2C', ['H2C', 'C2H'], "Which conversion model to run")

flags.DEFINE_bool('include_masks', True, "Is model conditioned on the ROIs.")

FLAGS = flags.FLAGS


def _make_dir_if_not_exists(dir_path):
    """Make a directory if it does not exist."""
    if not tf.gfile.Exists(dir_path):
        tf.gfile.MakeDirs(dir_path)


def make_inference_graph(model_name):
    """Build the inference graph for either the H2C or C2H GAN.

    Args:
      model_name: The var scope name 'ModelH2C' or 'ModelC2H'.

    Returns:
      Tuple of (input_placeholder, generated_tensor).
    """
    num_outputs = 2 if FLAGS.include_masks else 1
    if FLAGS.include_masks:
        input_pl = tf.placeholder(tf.float32, [FLAGS.height, FLAGS.width, 2])
        image, mask = tf.unstack(input_pl, axis=2)  # to HW
        image = tf.expand_dims(data_provider.normalize_image(image, [FLAGS.height, FLAGS.width]), 0)  # to NHWC
        mask = tf.expand_dims(data_provider.normalize_image(mask, [FLAGS.height, FLAGS.width]), 0)
        data_in = tf.concat([image, mask], axis=3)
    else:
        input_pl = tf.placeholder(tf.float32, [FLAGS.height, FLAGS.width])
        image = tf.expand_dims(data_provider.normalize_image(input_pl, [FLAGS.height, FLAGS.width]), 0)
        data_in = image

    # Expand HW to NHWC and normalize image between -1 and 1.

    with tf.variable_scope(model_name):
        with tf.variable_scope('Generator'):
            generated = generator.generator(
                    data_in, use_icnr=FLAGS.use_icnr, upsample_method=FLAGS.upsample_method, num_outputs=num_outputs)
    return input_pl, generated


def to_example_extended(img_path, img_orig, img_gen, img_cycle, mask_orig, mask_gen, mask_cycle, width, height,
                        label_orig, label_gen):
    assert img_orig.shape == mask_orig.shape
    assert img_orig.shape == img_gen.shape
    assert img_orig.shape == img_cycle.shape
    assert img_gen.shape == mask_gen.shape
    assert img_cycle.shape == mask_cycle.shape
    assert isinstance(label_orig, np.int64)
    assert isinstance(label_gen, np.int64)
    features_dict = features.to_feature_dict(img_path, img_orig, mask_orig, width, height, label_orig)
    # Add the generated data with the _gen suffix to the dict.
    features_dict.update(features.to_feature_dict(img_path, img_gen, mask_gen, width, height, label_gen, suffix="_gen"))
    features_dict.update(
            features.to_feature_dict(img_path, img_cycle, mask_cycle, width, height, label_orig, suffix="_cycle"))
    # Create an example protocol buffer
    return features.to_example(features_dict)


def export(sess, input_pls, output_tensors, input_file_path, output_dir):
    """Exports inference outputs to an output directory.

      Args:
        sess: tf.Session with variables already loaded.
        input_pl: tf.Placeholder for input (HWC format).
        output_tensor: Tensor for generated outut images.
        input_file_path: Input tfrecords file.
        output_dir: Output directory.
    """

    def get_outputs(output_image):
        if FLAGS.include_masks:
            output_image, output_mask = np.split(output_image[0, :, :, :2], 2, axis=2)
            output_image = data_provider.undo_normalize_image(output_image)
            output_mask = data_provider.undo_normalize_image(output_mask)
        else:
            output_image = np.squeeze(output_image)
            output_image = data_provider.undo_normalize_image(output_image)
            output_mask = np.zeros_like(output_image)
        return output_image, output_mask

    x2y_pl, y2x_pl = input_pls
    output_tensor_y, output_tensor_x = output_tensors

    if output_dir:
        _make_dir_if_not_exists(output_dir)

    if input_file_path:
        dir_name, file_name = os.path.split(input_file_path)
        file_name = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(FLAGS.generated_dir, file_name + '_gen.tfrecord')
        with tf.python_io.TFRecordWriter(output_file_path, options=OPTIONS) as writer:
            record_iterator = tf.python_io.tf_record_iterator(path=input_file_path, options=OPTIONS)
            for i, string_record in enumerate(record_iterator):
                print(i)
                concat, image, mask, label, image_path = data_provider.parse_example(
                        string_record, [FLAGS.height, FLAGS.width])
                concat, image, mask, label, image_path = sess.run([concat, image, mask, label, image_path])
                image_path = str(image_path)
                input_image = concat if FLAGS.include_masks else image
                output_image_y = sess.run(output_tensor_y, feed_dict={x2y_pl: input_image})
                output_image_y, output_mask_y = get_outputs(output_image_y)
                output_image_x = sess.run(output_tensor_x, feed_dict={y2x_pl: output_image_y})
                output_image_x, output_mask_x = get_outputs(output_image_x)
                label_gen = np.int64(not bool(label))  # Swap the label.
                proto = to_example_extended(image_path, image, output_image_y, output_image_x, mask, output_mask_y,
                                            output_mask_x, FLAGS.width, FLAGS.height, label, label_gen)
                writer.write(proto.SerializeToString())


def main(_):
    x2y = FLAGS.model
    y2x = FLAGS.model[::-1]  # reverse the string .. h2c->c2h
    images_x_hwc_pl, generated_y = make_inference_graph('Model{}'.format(x2y))
    images_y_hwc_pl, generated_x = make_inference_graph('Model{}'.format(y2x))

    # Restore all the variables that were saved in the checkpoint.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Restoring from", FLAGS.checkpoint_path)
        saver.restore(sess, FLAGS.checkpoint_path)
        print("Predicting '{}' --> '{}'".format(FLAGS.image_source, FLAGS.generated_dir))
        export(sess, (images_x_hwc_pl, images_y_hwc_pl), (generated_y, generated_x), FLAGS.image_source,
               FLAGS.generated_dir)


if __name__ == '__main__':
    tf.app.run()
