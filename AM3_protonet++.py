#!/usr/bin/env python3

"""Training and evaluation entry point."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
from scipy.spatial import KDTree
from common.util import Dataset
from common.util import ACTIVATION_MAP
from tqdm import trange
import pathlib
import logging
from common.util import summary_writer
from common.gen_experiments import load_and_save_params
import time
import pickle as pkl


tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)



def _load_mini_imagenet(data_dir, split):
    """Load mini-imagenet from numpy's npz file format."""
    _split_tag = {'sources': 'train', 'target_val': 'val', 'target_tst': 'test'}[split]
    dataset_path = os.path.join(data_dir, 'few-shot-{}.npz'.format(_split_tag))
    logging.info("Loading mini-imagenet...")
    data = np.load(dataset_path)
    fields = data['features'], data['targets']
    logging.info("Done loading.")
    return fields

def get_image_size(data_dir):
    if 'mini-imagenet' or 'tiered' in data_dir:
        image_size = 84
    elif 'cifar' in data_dir:
        image_size = 32
    else:
        raise Exception('Unknown dataset: %s' % data_dir)
    return image_size


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'test', 'train_classifier', 'create_embedding'])
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data.')
    parser.add_argument('--data_split', type=str, default='sources', choices=['sources', 'target_val', 'target_tst'],
                        help='Split of the data to be used to perform operation.')

    # Training parameters
    parser.add_argument('--number_of_steps', type=int, default=int(30000),
                        help="Number of training steps (number of Epochs in Hugo's paper)")
    parser.add_argument('--number_of_steps_to_early_stop', type=int, default=int(1000000),
                        help="Number of training steps after half way to early stop the training")
    parser.add_argument('--log_dir', type=str, default='', help='Base log dir')
    parser.add_argument('--num_classes_train', type=int, default=5,
                        help='Number of classes in the train phase, this is coming from the prototypical networks')
    parser.add_argument('--num_shots_train', type=int, default=5,
                        help='Number of shots in a few shot meta-train scenario')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--num_tasks_per_batch', type=int, default=2,
                        help='Number of few shot tasks per batch, so the task encoding batch is num_tasks_per_batch x num_classes_test x num_shots_train .')
    parser.add_argument('--init_learning_rate', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--save_summaries_secs', type=int, default=60, help='Time between saving summaries')
    parser.add_argument('--save_interval_secs', type=int, default=60, help='Time between saving model?')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--augment', type=bool, default=False)
    # Learning rate paramteres
    parser.add_argument('--lr_anneal', type=str, default='pwc', choices=['const', 'pwc', 'cos', 'exp'])
    parser.add_argument('--n_lr_decay', type=int, default=3)
    parser.add_argument('--lr_decay_rate', type=float, default=10.0)
    parser.add_argument('--num_steps_decay_pwc', type=int, default=2500,
                        help='Decay learning rate every num_steps_decay_pwc')

    parser.add_argument('--clip_gradient_norm', type=float, default=1.0, help='gradient clip norm.')
    parser.add_argument('--weights_initializer_factor', type=float, default=0.1,
                        help='multiplier in the variance of the initialization noise.')
    # Evaluation parameters
    parser.add_argument('--max_number_of_evaluations', type=float, default=float('inf'))
    parser.add_argument('--eval_interval_secs', type=int, default=120, help='Time between evaluating model?')
    parser.add_argument('--eval_interval_steps', type=int, default=1000,
                        help='Number of train steps between evaluating model in the training loop')
    parser.add_argument('--eval_interval_fine_steps', type=int, default=250,
                        help='Number of train steps between evaluating model in the training loop in the final phase')
    # Test parameters
    parser.add_argument('--num_classes_test', type=int, default=5, help='Number of classes in the test phase')
    parser.add_argument('--num_shots_test', type=int, default=5,
                        help='Number of shots in a few shot meta-test scenario')
    parser.add_argument('--num_cases_test', type=int, default=100000,
                        help='Number of few-shot cases to compute test accuracy')
    # Architecture parameters
    parser.add_argument('--dropout', type=float, default=1.0)
    parser.add_argument('--conv_dropout', type=float, default=None)
    parser.add_argument('--feature_dropout_p', type=float, default=None)

    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--num_units_in_block', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_max_pools', type=int, default=3)
    parser.add_argument('--block_size_growth', type=float, default=2.0)
    parser.add_argument('--activation', type=str, default='swish-1', choices=['relu', 'selu', 'swish-1'])

    parser.add_argument('--feature_expansion_size', type=int, default=None)
    parser.add_argument('--feature_bottleneck_size', type=int, default=None)

    parser.add_argument('--feature_extractor', type=str, default='simple_res_net',
                        choices=['simple_res_net'], help='Which feature extractor to use')


    parser.add_argument('--encoder_sharing', type=str, default='shared',
                        choices=['shared'],
                        help='How to link fetaure extractors in task encoder and classifier')
    parser.add_argument('--encoder_classifier_link', type=str, default='prototypical',
                        choices=['prototypical'],
                        help='How to link fetaure extractors in task encoder and classifier')
    parser.add_argument('--embedding_pooled', type=bool, default=True,
                        help='Whether to use avg pooling to create embedding')
    parser.add_argument('--task_encoder', type=str, default='self_att_mlp',
                        choices=['class_mean', 'fixed_alpha','fixed_alpha_mlp','self_att_mlp'])

    #
    parser.add_argument('--num_batches_neg_mining', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=100, help='Evaluation batch size')

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--mlp_weight_decay', type=float, default=0.0)
    parser.add_argument('--mlp_dropout', type=float, default=0.0)
    parser.add_argument('--mlp_type', type=str, default='non-linear')
    parser.add_argument('--att_input', type=str, default='word')

    args = parser.parse_args()

    print(args)
    return args


def get_logdir_name(flags):
    """Generates the name of the log directory from the values of flags
    Parameters
    ----------
        flags: neural net architecture generated by get_arguments()
    Outputs
    -------
        the name of the directory to store the training and evaluation results
    """
    logdir = flags.log_dir

    return logdir


class ScaledVarianceRandomNormal(init_ops.Initializer):
    """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, factor=1.0, seed=None, dtype=dtypes.float32):
        self.mean = mean
        self.factor = factor
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if shape:
            n = float(shape[-1])
        else:
            n = 1.0
        for dim in shape[:-2]:
            n *= float(dim)

        self.stddev = np.sqrt(self.factor * 2.0 / n)
        return random_ops.random_normal(shape, self.mean, self.stddev,
                                        dtype, seed=self.seed)


def _get_scope(is_training, flags):
    normalizer_params = {
        'epsilon': 0.001,
        'momentum': .95,
        'trainable': is_training,
        'training': is_training,
    }
    conv2d_arg_scope = slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params=normalizer_params,
        # padding='SAME',
        trainable=is_training,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
        weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
        biases_initializer=tf.constant_initializer(0.0)
    )
    dropout_arg_scope = slim.arg_scope(
        [slim.dropout],
        keep_prob=flags.dropout,
        is_training=is_training)
    return conv2d_arg_scope, dropout_arg_scope


def build_simple_conv_net(images, flags, is_training, reuse=None, scope=None):
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    with conv2d_arg_scope, dropout_arg_scope:
        with tf.variable_scope(scope or 'feature_extractor', reuse=reuse):
            h = images
            for i in range(4):
                h = slim.conv2d(h, num_outputs=flags.num_filters, kernel_size=3, stride=1,
                                scope='conv' + str(i), padding='SAME',
                                weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor))
                h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='VALID', scope='max_pool' + str(i))

            if flags.embedding_pooled == True:
                kernel_size = h.shape.as_list()[-2]
                h = slim.avg_pool2d(h, kernel_size=kernel_size, scope='avg_pool')
            h = slim.flatten(h)
    return h


def leaky_relu(x, alpha=0.1, name=None):
    return tf.maximum(x, alpha * x, name=name)




def build_simple_res_net(images, flags, num_filters, beta=None, gamma=None, is_training=False, reuse=None, scope=None):
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    activation_fn = ACTIVATION_MAP[flags.activation]
    with conv2d_arg_scope, dropout_arg_scope:
        with tf.variable_scope(scope or 'feature_extractor', reuse=reuse):
            h = images
            for i in range(len(num_filters)):
                # make shortcut
                shortcut = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=1, stride=1,
                                       activation_fn=None,
                                       scope='shortcut' + str(i), padding='SAME')

                for j in range(flags.num_units_in_block):
                    h = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=3, stride=1,
                                    scope='conv' + str(i) + '_' + str(j), padding='SAME', activation_fn=None)
                    if flags.conv_dropout:
                        h = slim.dropout(h, keep_prob=1.0 - flags.conv_dropout)

                    if j < (flags.num_units_in_block - 1):
                        h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))
                h = h + shortcut

                h = activation_fn(h, name='activation_' + str(i) + '_' + str(flags.num_units_in_block - 1))
                if i < flags.num_max_pools:
                    h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool' + str(i))

            if flags.feature_expansion_size:
                if flags.feature_dropout_p:
                    h = slim.dropout(h, scope='feature_expansion_dropout', keep_prob=1.0 - flags.feature_dropout_p)
                h = slim.conv2d(slim.dropout(h), num_outputs=flags.feature_expansion_size, kernel_size=1, stride=1,
                                scope='feature_expansion', padding='SAME')

            if flags.embedding_pooled == True:
                kernel_size = h.shape.as_list()[-2]
                h = slim.avg_pool2d(h, kernel_size=kernel_size, scope='avg_pool')
            h = slim.flatten(h)

            if flags.feature_dropout_p:
                h = slim.dropout(h, scope='feature_bottleneck_dropout', keep_prob=1.0 - flags.feature_dropout_p)
            # Bottleneck layer
            if flags.feature_bottleneck_size:
                h = slim.fully_connected(h, num_outputs=flags.feature_bottleneck_size,
                                         activation_fn=activation_fn, normalizer_fn=None,
                                         scope='feature_bottleneck')

    return h



def build_wordemb_transformer(embeddings, flags, is_training=False, reuse=None, scope=None):
    with tf.variable_scope(scope or 'mlp_transformer', reuse=reuse):
        h = embeddings
        if flags.mlp_type=='linear':
            h = slim.fully_connected(h, 512, reuse=False, scope='mlp_layer',
                                     activation_fn=None, trainable=is_training,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.mlp_weight_decay),
                                     weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
                                     biases_initializer=tf.constant_initializer(0.0))
        elif flags.mlp_type=='non-linear':
            h = slim.fully_connected(h, 300, reuse=False, scope='mlp_layer',
                                     activation_fn=tf.nn.relu, trainable=is_training,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(
                                         scale=flags.mlp_weight_decay),
                                     weights_initializer=ScaledVarianceRandomNormal(
                                         factor=flags.weights_initializer_factor),
                                     biases_initializer=tf.constant_initializer(0.0))
            h = slim.dropout(h, scope='mlp_dropout', keep_prob=1.0 - flags.mlp_dropout, is_training=is_training)
            h = slim.fully_connected(h, 512, reuse=False, scope='mlp_layer_1',
                                     activation_fn=None, trainable=is_training,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(
                                         scale=flags.mlp_weight_decay),
                                     weights_initializer=ScaledVarianceRandomNormal(
                                         factor=flags.weights_initializer_factor),
                                     biases_initializer=tf.constant_initializer(0.0))

    return h

def build_self_attention(embeddings, flags, is_training=False, reuse=None, scope=None):
    with tf.variable_scope(scope or 'self_attention', reuse=reuse):
        h = embeddings
        if flags.mlp_type=='linear':
            h = slim.fully_connected(h, 1, reuse=False, scope='self_att_layer',
                                     activation_fn=None, trainable=is_training,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.mlp_weight_decay),
                                     weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
                                     biases_initializer=tf.constant_initializer(0.0))
        elif flags.mlp_type=='non-linear':
            h = slim.fully_connected(h, 300, reuse=False, scope='self_att_layer',
                                     activation_fn=tf.nn.relu, trainable=is_training,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(
                                         scale=flags.mlp_weight_decay),
                                     weights_initializer=ScaledVarianceRandomNormal(
                                         factor=flags.weights_initializer_factor),
                                     biases_initializer=tf.constant_initializer(0.0))
            h = slim.dropout(h, scope='self_att_dropout', keep_prob=1.0 - flags.mlp_dropout, is_training=is_training)
            h = slim.fully_connected(h, 1, reuse=False, scope='self_att_layer_1',
                                     activation_fn=None, trainable=is_training,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(
                                         scale=flags.mlp_weight_decay),
                                     weights_initializer=ScaledVarianceRandomNormal(
                                         factor=flags.weights_initializer_factor),
                                     biases_initializer=tf.constant_initializer(0.0))
        h = tf.sigmoid(h)

    return h

def get_res_net_block(h, flags, num_filters, num_units, pool=False, is_training=False,
                      reuse=None, scope=None):
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    activation_fn = ACTIVATION_MAP[flags.activation]
    with conv2d_arg_scope, dropout_arg_scope:
        with tf.variable_scope(scope, reuse=reuse):
            # make shortcut
            shortcut = slim.conv2d(h, num_outputs=num_filters, kernel_size=1, stride=1,
                                   activation_fn=None,
                                   scope='shortcut', padding='SAME')

            for j in range(num_units):
                h = slim.conv2d(h, num_outputs=num_filters, kernel_size=3, stride=1,
                                scope='conv_' + str(j), padding='SAME', activation_fn=None)
                if flags.conv_dropout:
                    h = slim.dropout(h, keep_prob=1.0 - flags.conv_dropout)
                if j < (num_units - 1):
                    h = activation_fn(h, name='activation_' + str(j))
            h = h + shortcut
            h = activation_fn(h, name='activation_' + '_' + str(flags.num_units_in_block - 1))
            if pool:
                h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool')
    return h



def build_feature_extractor_graph(images, flags, num_filters, beta=None, gamma=None, is_training=False,
                                  scope='feature_extractor_task_encoder', reuse=None, is_64way=False):
    if flags.feature_extractor == 'simple_conv_net':
        h = build_simple_conv_net(images, flags=flags, is_training=is_training, reuse=reuse, scope=scope)
    elif flags.feature_extractor == 'simple_res_net':
        h = build_simple_res_net(images, flags=flags, num_filters=num_filters, beta=beta, gamma=gamma,
                                 is_training=is_training, reuse=reuse, scope=scope)
    else:
        h = None

    embedding_shape = h.get_shape().as_list()
    if is_training and is_64way is False:
        h = tf.reshape(h, shape=(flags.num_tasks_per_batch, embedding_shape[0] // flags.num_tasks_per_batch, -1),
                       name='reshape_to_separate_tasks_generic_features')
    else:
        h = tf.reshape(h, shape=(1, embedding_shape[0], -1),
                       name='reshape_to_separate_tasks_generic_features')

    return h



def build_task_encoder(embeddings, label_embeddings, flags, is_training, querys=None, reuse=None, scope='class_encoder'):
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    alpha=None

    with conv2d_arg_scope, dropout_arg_scope:
        with tf.variable_scope(scope, reuse=reuse):

            if flags.task_encoder == 'talkthrough':
                task_encoding = embeddings
            elif flags.task_encoder == 'class_mean':
                task_encoding = embeddings

                if is_training:
                    task_encoding = tf.reshape(task_encoding, shape=(
                    flags.num_tasks_per_batch, flags.num_classes_train, flags.num_shots_train, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                else:
                    task_encoding = tf.reshape(task_encoding,
                                               shape=(1, flags.num_classes_test, flags.num_shots_test, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                task_encoding = tf.reduce_mean(task_encoding, axis=2, keep_dims=False)
            elif flags.task_encoder == 'fixed_alpha':
                task_encoding = embeddings
                print("entered the word embedding task encoder...")

                if is_training:
                    task_encoding = tf.reshape(task_encoding, shape=(
                        flags.num_tasks_per_batch, flags.num_classes_train, flags.num_shots_train, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                    label_embeddings = tf.reshape(label_embeddings, shape=(
                        flags.num_tasks_per_batch, flags.num_classes_train, -1),
                                               name='reshape_to_separate_tasks_label_embedding')
                else:
                    task_encoding = tf.reshape(task_encoding,
                                               shape=(1, flags.num_classes_test, flags.num_shots_test, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                    label_embeddings = tf.reshape(label_embeddings,
                                               shape=(1, flags.num_classes_test, -1),
                                               name='reshape_to_separate_tasks_label_embedding')
                task_encoding = tf.reduce_mean(task_encoding, axis=2, keep_dims=False)
                task_encoding = flags.alpha*task_encoding+(1-flags.alpha)*label_embeddings
            elif flags.task_encoder == 'fixed_alpha_mlp':
                task_encoding = embeddings
                print("entered the word embedding task encoder...")
                label_embeddings = build_wordemb_transformer(label_embeddings,flags,is_training)

                if is_training:
                    task_encoding = tf.reshape(task_encoding, shape=(
                        flags.num_tasks_per_batch, flags.num_classes_train, flags.num_shots_train, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                    label_embeddings = tf.reshape(label_embeddings, shape=(
                        flags.num_tasks_per_batch, flags.num_classes_train, -1),
                                               name='reshape_to_separate_tasks_label_embedding')
                else:
                    task_encoding = tf.reshape(task_encoding,
                                               shape=(1, flags.num_classes_test, flags.num_shots_test, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                    label_embeddings = tf.reshape(label_embeddings,
                                               shape=(1, flags.num_classes_test, -1),
                                               name='reshape_to_separate_tasks_label_embedding')
                task_encoding = tf.reduce_mean(task_encoding, axis=2, keep_dims=False)
                task_encoding = flags.alpha*task_encoding+(1-flags.alpha)*label_embeddings
            elif flags.task_encoder == 'self_att_mlp':
                task_encoding = embeddings
                print("entered the word embedding task encoder...")
                label_embeddings = build_wordemb_transformer(label_embeddings,flags,is_training)

                if is_training:
                    task_encoding = tf.reshape(task_encoding, shape=(
                        flags.num_tasks_per_batch, flags.num_classes_train, flags.num_shots_train, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                    label_embeddings = tf.reshape(label_embeddings, shape=(
                        flags.num_tasks_per_batch, flags.num_classes_train, -1),
                                               name='reshape_to_separate_tasks_label_embedding')
                else:
                    task_encoding = tf.reshape(task_encoding,
                                               shape=(1, flags.num_classes_test, flags.num_shots_test, -1),
                                               name='reshape_to_separate_tasks_task_encoding')
                    label_embeddings = tf.reshape(label_embeddings,
                                               shape=(1, flags.num_classes_test, -1),
                                               name='reshape_to_separate_tasks_label_embedding')
                task_encoding = tf.reduce_mean(task_encoding, axis=2, keep_dims=False)

                if flags.att_input=='proto':
                    alpha = build_self_attention(task_encoding,flags,is_training)
                elif flags.att_input=='word':
                    alpha = build_self_attention(label_embeddings,flags,is_training)
                elif flags.att_input=='combined':
                    embeddings=tf.concat([task_encoding, label_embeddings], axis=2)
                    alpha = build_self_attention(embeddings, flags, is_training)

                elif flags.att_input=='queryword':
                    j = label_embeddings.get_shape().as_list()[1]
                    i = querys.get_shape().as_list()[1]
                    task_encoding_tile = tf.expand_dims(task_encoding, axis=1)
                    task_encoding_tile = tf.tile(task_encoding_tile, (1, i, 1, 1))
                    querys_tile = tf.expand_dims(querys, axis=2)
                    querys_tile = tf.tile(querys_tile, (1, 1, j, 1))
                    label_embeddings_tile = tf.expand_dims(label_embeddings, axis=1)
                    label_embeddings_tile = tf.tile(label_embeddings_tile, (1, i, 1, 1))
                    att_input = tf.concat([label_embeddings_tile, querys_tile], axis=3)
                    alpha = build_self_attention(att_input, flags, is_training)
                elif flags.att_input=='queryproto':
                    j = task_encoding.get_shape().as_list()[1]
                    i = querys.get_shape().as_list()[1]
                    task_encoding_tile = tf.expand_dims(task_encoding, axis=1)
                    task_encoding_tile = tf.tile(task_encoding_tile, (1, i, 1, 1))
                    querys_tile = tf.expand_dims(querys, axis=2)
                    querys_tile = tf.tile(querys_tile, (1, 1, j, 1))
                    label_embeddings_tile = tf.expand_dims(label_embeddings, axis=1)
                    label_embeddings_tile = tf.tile(label_embeddings_tile, (1, i, 1, 1))
                    att_input = tf.concat([task_encoding_tile, querys_tile], axis=3)
                    alpha = build_self_attention(att_input, flags, is_training)

                if querys is None:
                    task_encoding = alpha*task_encoding+(1-alpha)*label_embeddings
                else:
                    task_encoding = alpha * task_encoding_tile + (1-alpha) * label_embeddings_tile

            else:
                task_encoding = None

            return task_encoding, alpha


def build_prototypical_head(features_generic, task_encoding, flags, is_training, scope='prototypical_head'):
    """
    Implements the prototypical networks few-shot head
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):

        if len(features_generic.get_shape().as_list()) == 2:
            features_generic = tf.expand_dims(features_generic, axis=0)
        if len(task_encoding.get_shape().as_list()) == 2:
            task_encoding = tf.expand_dims(task_encoding, axis=0)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[1]
        i = features_generic.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        features_generic = tf.expand_dims(features_generic, axis=2)
        task_encoding = tf.expand_dims(task_encoding, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        task_encoding_tile = tf.tile(task_encoding, (1, i, 1, 1))
        features_generic_tile = tf.tile(features_generic, (1, 1, j, 1))
        # implement equation (4)
        euclidian = -tf.norm(task_encoding_tile - features_generic_tile, name='neg_euclidian_distance', axis=-1)

        if is_training:
            euclidian = tf.reshape(euclidian, shape=(flags.num_tasks_per_batch * flags.train_batch_size, -1))
        else:
            euclidian_shape = euclidian.get_shape().as_list()
            euclidian = tf.reshape(euclidian, shape=(euclidian_shape[1], -1))

        return euclidian


def build_prototypical_head_protoperquery(features_generic, task_encoding, flags, is_training, scope='prototypical_head'):
    """
    Implements the prototypical networks few-shot head
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """
    # the shape of task_encoding is [num_tasks, batch_size, num_classes, ]

    with tf.variable_scope(scope):

        if len(features_generic.get_shape().as_list()) == 2:
            features_generic = tf.expand_dims(features_generic, axis=0)
        if len(task_encoding.get_shape().as_list()) == 2:
            task_encoding = tf.expand_dims(task_encoding, axis=0)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[2]
        i = features_generic.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        features_generic = tf.expand_dims(features_generic, axis=2)
        #task_encoding = tf.expand_dims(task_encoding, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        features_generic_tile = tf.tile(features_generic, (1, 1, j, 1))
        # implement equation (4)
        euclidian = -tf.norm(task_encoding - features_generic_tile, name='neg_euclidian_distance', axis=-1)

        if is_training:
            euclidian = tf.reshape(euclidian, shape=(flags.num_tasks_per_batch * flags.train_batch_size, -1))
        else:
            euclidian_shape = euclidian.get_shape().as_list()
            euclidian = tf.reshape(euclidian, shape=(euclidian_shape[1], -1))

        return euclidian

def build_regularizer_head(embeddings, label_embeddings, flags, is_training, scope='regularizer_head'):
    """
    Implements the prototypical networks few-shot head
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):
        task_encoding = embeddings

        if is_training:
            task_encoding = tf.reshape(task_encoding, shape=(
                flags.num_tasks_per_batch, flags.num_classes_train, flags.num_shots_train, -1),
                                       name='reshape_to_separate_tasks_task_encoding')
            label_embeddings = tf.reshape(label_embeddings, shape=(
                flags.num_tasks_per_batch, flags.num_classes_train, -1),
                                          name='reshape_to_separate_tasks_label_embedding')
        else:
            task_encoding = tf.reshape(task_encoding,
                                       shape=(1, flags.num_classes_test, flags.num_shots_test, -1),
                                       name='reshape_to_separate_tasks_task_encoding')
            label_embeddings = tf.reshape(label_embeddings,
                                          shape=(1, flags.num_classes_test, -1),
                                          name='reshape_to_separate_tasks_label_embedding')
        task_encoding = tf.reduce_mean(task_encoding, axis=2, keep_dims=False)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[1]
        i = label_embeddings.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        task_encoding = tf.expand_dims(task_encoding, axis=2)
        label_embeddings = tf.expand_dims(label_embeddings, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        label_embeddings_tile = tf.tile(label_embeddings, (1, i, 1, 1))
        task_encoding_tile = tf.tile(task_encoding, (1, 1, j, 1))
        # implement equation (4)
        euclidian = -tf.norm(task_encoding_tile - label_embeddings_tile, name='neg_euclidian_distance_regularizer', axis=-1)

        if is_training:
            euclidian = tf.reshape(euclidian, shape=(flags.num_tasks_per_batch * flags.num_classes_train, -1))
        else:
            euclidian_shape = euclidian.get_shape().as_list()
            euclidian = tf.reshape(euclidian, shape=(euclidian_shape[1], -1))

        return euclidian


def placeholder_inputs(batch_size, image_size, scope):
    """
    :param batch_size:
    :return: placeholders for images and
    """
    with tf.variable_scope(scope):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='images')
        labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size), name='labels')
        return images_placeholder, labels_placeholder


def get_batch(data_set, images_placeholder, labels_placeholder, batch_size):
    """
    :param data_set:
    :param images_placeholder:
    :param labels_placeholder:
    :return:
    """
    images_feed, labels_feed = data_set.next_batch(batch_size)

    feed_dict = {
        images_placeholder: images_feed.astype(dtype=np.float32),
        labels_placeholder: labels_feed,
    }
    return feed_dict


def preprocess(images):
    # mean = tf.constant(np.asarray([127.5, 127.5, 127.5]).reshape([1, 1, 3]), dtype=tf.float32, name='image_mean')
    # std = tf.constant(np.asarray([127.5, 127.5, 127.5]).reshape([1, 1, 3]), dtype=tf.float32, name='image_std')
    # return tf.div(tf.subtract(images, mean), std)

    std = tf.constant(np.asarray([0.5, 0.5, 0.5]).reshape([1, 1, 3]), dtype=tf.float32, name='image_std')
    return tf.div(images, std)


def get_nearest_neighbour_acc(flags, embeddings, labels):
    num_correct = 0
    num_tot = 0
    for i in trange(flags.num_cases_test):
        test_classes = np.random.choice(np.unique(labels), size=flags.num_classes_test, replace=False)
        train_idxs, test_idxs = get_few_shot_idxs(labels=labels, classes=test_classes, num_shots=flags.num_shots_test)
        # TODO: this is to fix the OOM error, this can be removed when embed() supports batch processing
        test_idxs = np.random.choice(test_idxs, size=100, replace=False)

        np_embedding_train = embeddings[train_idxs]
        # Using the np.std instead of np.linalg.norm improves results by around 1-1.5%
        np_embedding_train = np_embedding_train / np.std(np_embedding_train, axis=1, keepdims=True)
        # np_embedding_train = np_embedding_train / np.linalg.norm(np_embedding_train, axis=1, keepdims=True)
        labels_train = labels[train_idxs]

        np_embedding_test = embeddings[test_idxs]
        np_embedding_test = np_embedding_test / np.std(np_embedding_test, axis=1, keepdims=True)
        # np_embedding_test = np_embedding_test / np.linalg.norm(np_embedding_test, axis=1, keepdims=True)
        labels_test = labels[test_idxs]

        kdtree = KDTree(np_embedding_train)
        nns, nn_idxs = kdtree.query(np_embedding_test, k=1)
        labels_predicted = labels_train[nn_idxs]

        num_matches = sum(labels_predicted == labels_test)

        num_correct += num_matches
        num_tot += len(labels_predicted)

    # print("Accuracy: ", (100.0 * num_correct) / num_tot)
    return (100.0 * num_correct) / num_tot



def build_inference_graph(images_deploy_pl, images_task_encode_pl, flags, is_training,
                          is_primary, label_embeddings):
    num_filters = [round(flags.num_filters * pow(flags.block_size_growth, i)) for i in range(flags.num_blocks)]
    reuse = not is_primary
    alpha=None

    with tf.variable_scope('Model'):
        feature_extractor_encoding_scope = 'feature_extractor_encoder'

        features_task_encode = build_feature_extractor_graph(images=images_task_encode_pl, flags=flags,
                                                             is_training=is_training,
                                                             num_filters=num_filters,
                                                             scope=feature_extractor_encoding_scope,
                                                             reuse=False)
        if flags.encoder_sharing == 'shared':
            ecoder_reuse = True
            feature_extractor_classifier_scope = feature_extractor_encoding_scope
        elif flags.encoder_sharing == 'siamese':
            # TODO: in the case of pretrained feature extractor this is not good,
            # because the classfier part will be randomly initialized
            ecoder_reuse = False
            feature_extractor_classifier_scope = 'feature_extractor_classifier'
        else:
            raise Exception('Option not implemented')

        if flags.encoder_classifier_link == 'prototypical':
            #flags.task_encoder = 'class_mean'
            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             scope=feature_extractor_classifier_scope,
                                                             num_filters=num_filters,
                                                             reuse=ecoder_reuse)
            querys = None
            if 'query' in flags.att_input:
                querys = features_generic
            task_encoding, alpha = build_task_encoder(embeddings=features_task_encode,
                                                      label_embeddings=label_embeddings,
                                                      flags=flags, is_training=is_training, reuse=reuse, querys=querys,
                                                      threshold=flags.alpha)
            if 'query' in flags.att_input:
                logits = build_prototypical_head_protoperquery(features_generic, task_encoding, flags,
                                                               is_training=is_training)
            else:
                logits = build_prototypical_head(features_generic, task_encoding, flags, is_training=is_training)
            # logits_regularizer = build_regularizer_head(embeddings= features_task_encode,
            #                                             label_embeddings=label_embeddings, flags=flags,
            #                                             is_training=is_training )
        else:
            raise Exception('Option not implemented')

    return logits, None, features_task_encode, features_generic, alpha




def get_train_datasets(flags):
    mini_imagenet = _load_mini_imagenet(data_dir=flags.data_dir, split='sources')
    few_shot_data_train = Dataset(mini_imagenet)
    pretrain_data_train, pretrain_data_test = None, None
    return few_shot_data_train, pretrain_data_train, pretrain_data_test


def get_pwc_learning_rate(global_step, flags):
    learning_rate = tf.train.piecewise_constant(global_step, [np.int64(flags.number_of_steps / 2),
                                                              np.int64(
                                                                  flags.number_of_steps / 2 + flags.num_steps_decay_pwc),
                                                              np.int64(
                                                                  flags.number_of_steps / 2 + 2 * flags.num_steps_decay_pwc)],
                                                [flags.init_learning_rate, flags.init_learning_rate * 0.1,
                                                 flags.init_learning_rate * 0.01,
                                                 flags.init_learning_rate * 0.001])
    return learning_rate


def create_hard_negative_batch(misclass, feed_dict, sess, few_shot_data_train, flags,
                               images_deploy_pl, labels_deploy_pl, images_task_encode_pl, labels_task_encode_pl):
    """

    :param logits:
    :param feed_dict:
    :param sess:
    :param few_shot_data_train:
    :param flags:
    :param images_deploy_pl:
    :param labels_deploy_pl:
    :param images_task_encode_pl:
    :param labels_task_encode_pl:
    :return:
    """
    feed_dict_test = dict(feed_dict)
    misclass_test_final = 0.0
    misclass_history = np.zeros(flags.num_batches_neg_mining)
    for i in range(flags.num_batches_neg_mining):
        images_deploy, labels_deploy, images_task_encode, labels_task_encode = \
            few_shot_data_train.next_few_shot_batch(deploy_batch_size=flags.train_batch_size,
                                                    num_classes_test=flags.num_classes_train,
                                                    num_shots=flags.num_shots_train,
                                                    num_tasks=flags.num_tasks_per_batch)

        feed_dict_test[images_deploy_pl] = images_deploy.astype(dtype=np.float32)
        feed_dict_test[labels_deploy_pl] = labels_deploy
        feed_dict_test[images_task_encode_pl] = images_task_encode.astype(dtype=np.float32)
        feed_dict_test[labels_task_encode_pl] = labels_task_encode

        # logits
        misclass_test = sess.run(misclass, feed_dict=feed_dict_test)
        misclass_history[i] = misclass_test
        if misclass_test > misclass_test_final:
            misclass_test_final = misclass_test
            feed_dict = dict(feed_dict_test)

    return feed_dict


def train(flags):
    log_dir = get_logdir_name(flags)
    flags.pretrained_model_dir = log_dir
    fout=open(log_dir+'/out','a')
    log_dir = os.path.join(log_dir, 'train')
    # This is setting to run evaluation loop only once
    flags.max_number_of_evaluations = 1
    flags.eval_interval_secs = 0
    image_size = get_image_size(flags.data_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        global_step_pretrain = tf.Variable(0, trainable=False, name='global_step_pretrain', dtype=tf.int64)

        images_deploy_pl, labels_deploy_pl = placeholder_inputs(
            batch_size=flags.num_tasks_per_batch * flags.train_batch_size,
            image_size=image_size, scope='inputs/deploy')
        images_task_encode_pl, _ = placeholder_inputs(
            batch_size=flags.num_tasks_per_batch * flags.num_classes_train * flags.num_shots_train,
            image_size=image_size, scope='inputs/task_encode')
        with tf.variable_scope('inputs/task_encode'):
            labels_task_encode_pl_real = tf.placeholder(tf.int64,
                                         shape=(flags.num_tasks_per_batch * flags.num_classes_train), name='labels_real')
            labels_task_encode_pl = tf.placeholder(tf.int64,
                                                        shape=(flags.num_tasks_per_batch * flags.num_classes_train),
                                                        name='labels')

        #here is the word embedding layer for training

        emb_path = os.path.join(flags.data_dir, 'few-shot-wordemb-{}.npz'.format("train"))
        embedding_train = np.load(emb_path)["features"].astype(np.float32)
        print(embedding_train.dtype)
        logging.info("Loading mini-imagenet...")
        W_train = tf.constant(embedding_train, name="W_train")
        label_embeddings_train = tf.nn.embedding_lookup(W_train, labels_task_encode_pl_real)

        # Primary task operations
        logits, regularizer_logits, _, _, alpha = build_inference_graph(images_deploy_pl=images_deploy_pl,
                                             images_task_encode_pl=images_task_encode_pl,
                                             flags=flags, is_training=True, is_primary=True,
                                             label_embeddings=label_embeddings_train)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=tf.one_hot(labels_deploy_pl, flags.num_classes_train)))
        # Losses and optimizer
        regu_losses = slim.losses.get_regularization_losses()
        loss = tf.add_n([loss] + regu_losses)
        misclass = 1.0 - slim.metrics.accuracy(tf.argmax(logits, 1), labels_deploy_pl)

        # Learning rate
        if flags.lr_anneal == 'const':
            learning_rate = flags.init_learning_rate
        elif flags.lr_anneal == 'pwc':
            learning_rate = get_pwc_learning_rate(global_step, flags)
        elif flags.lr_anneal == 'exp':
            lr_decay_step = flags.number_of_steps // flags.n_lr_decay
            learning_rate = tf.train.exponential_decay(flags.init_learning_rate, global_step, lr_decay_step,
                                                       1.0 / flags.lr_decay_rate, staircase=True)
        else:
            raise Exception('Not implemented')

        # Optimizer
        if flags.optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step,
                                                 clip_gradient_norm=flags.clip_gradient_norm)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('misclassification', misclass)
        tf.summary.scalar('learning_rate', learning_rate)
        # Merge all summaries except for pretrain
        summary = tf.summary.merge(tf.get_collection('summaries', scope='(?!pretrain).*'))


        # Get datasets
        few_shot_data_train, pretrain_data_train, pretrain_data_test = get_train_datasets(flags)
        # Define session and logging
        summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        supervisor = tf.train.Supervisor(logdir=log_dir, init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=summary_writer,
                                         saver=saver,
                                         global_step=global_step, save_summaries_secs=flags.save_summaries_secs,
                                         save_model_secs=0)  # flags.save_interval_secs

        with supervisor.managed_session() as sess:
            checkpoint_step = sess.run(global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1

            eval_interval_steps = flags.eval_interval_steps
            for step in range(checkpoint_step, flags.number_of_steps):
                # get batch of data to compute classification loss
                images_deploy, labels_deploy, images_task_encode, labels_task_encode_real, labels_task_encode = \
                    few_shot_data_train.next_few_shot_batch_wordemb(deploy_batch_size=flags.train_batch_size,
                                                            num_classes_test=flags.num_classes_train,
                                                            num_shots=flags.num_shots_train,
                                                            num_tasks=flags.num_tasks_per_batch)
                if flags.augment:
                    images_deploy = image_augment(images_deploy)
                    images_task_encode = image_augment(images_task_encode)

                feed_dict = {images_deploy_pl: images_deploy.astype(dtype=np.float32), labels_deploy_pl: labels_deploy,
                             images_task_encode_pl: images_task_encode.astype(dtype=np.float32),
                             labels_task_encode_pl_real: labels_task_encode_real,
                             labels_task_encode_pl: labels_task_encode}


                t_batch = time.time()
                feed_dict = create_hard_negative_batch(misclass, feed_dict, sess, few_shot_data_train, flags,
                                                       images_deploy_pl, labels_deploy_pl, images_task_encode_pl,
                                                       labels_task_encode_pl_real)
                dt_batch = time.time() - t_batch

                t_train = time.time()
                loss,alpha_np = sess.run([train_op,alpha], feed_dict=feed_dict)
                dt_train = time.time() - t_train

                if step % 100 == 0:
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    logging.info("step %d, loss : %.4g, dt: %.3gs, dt_batch: %.3gs" % (step, loss, dt_train, dt_batch))
                    fout.write("step: "+str(step)+' loss: '+str(loss)+'\n')

                if float(step) / flags.number_of_steps > 0.5:
                    eval_interval_steps = flags.eval_interval_fine_steps

                if eval_interval_steps > 0 and step % eval_interval_steps == 0:
                    saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                    eval(flags=flags, is_primary=True, fout=fout)

                if float(step) > 0.5 * flags.number_of_steps + flags.number_of_steps_to_early_stop:
                    break



class ModelLoader:
    def __init__(self, model_path, batch_size, is_primary, split):
        self.batch_size = batch_size

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        image_size = get_image_size(flags.data_dir)

        with tf.Graph().as_default():
            images_deploy_pl, labels_deploy_pl = placeholder_inputs(batch_size=batch_size,
                                                                    image_size=image_size, scope='inputs/deploy')
            if is_primary:
                task_encode_batch_size = flags.num_classes_test * flags.num_shots_test
            images_task_encode_pl, _ = placeholder_inputs(batch_size=task_encode_batch_size,
                                                                              image_size=image_size,
                                                                              scope='inputs/task_encode')
            with tf.variable_scope('inputs/task_encode'):
                labels_task_encode_pl_real = tf.placeholder(tf.int64,
                                             shape=(flags.num_classes_test), name='labels_real')
                labels_task_encode_pl = tf.placeholder(tf.int64,
                                                       shape=(flags.num_classes_test),
                                                       name='labels')
                self.vocab_size = tf.placeholder(tf.float32, shape=(), name='vocab_size')
            self.tensor_images_deploy = images_deploy_pl
            self.tensor_labels_deploy = labels_deploy_pl
            self.tensor_labels_task_encode_real = labels_task_encode_pl_real
            self.tensor_labels_task_encode = labels_task_encode_pl
            self.tensor_images_task_encode = images_task_encode_pl

            emb_path = os.path.join(flags.data_dir, 'few-shot-wordemb-{}.npz'.format(split))
            embedding_train = np.load(emb_path)["features"].astype(np.float32)
            print(embedding_train.dtype)
            logging.info("Loading mini-imagenet...")
            W = tf.constant(embedding_train, name="W_"+split)


            label_embeddings_train = tf.nn.embedding_lookup(W, labels_task_encode_pl_real)

            # Primary task operations
            logits, regularizer_logits, features_sample, features_query, self.alpha = build_inference_graph(images_deploy_pl=images_deploy_pl,
                                                                     images_task_encode_pl=images_task_encode_pl,
                                                                     flags=flags, is_training=False, is_primary=True,
                                                                     label_embeddings=label_embeddings_train)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=tf.one_hot(labels_deploy_pl, flags.num_classes_test)))
            regularizer_loss = 0.0

            # Losses and optimizer
            regu_losses = slim.losses.get_regularization_losses()

            loss = tf.add_n([loss] + regu_losses + [regularizer_loss])

            init_fn = slim.assign_from_checkpoint_fn(
                latest_checkpoint,
                slim.get_model_variables('Model'))

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Run init before loading the weights
            self.sess.run(tf.global_variables_initializer())
            # Load weights
            init_fn(self.sess)

            self.flags = flags
            self.logits = logits
            self.loss = loss
            self.features_sample = features_sample
            self.features_query = features_query
            self.logits_size = self.logits.get_shape().as_list()[-1]
            self.step = step
            self.is_primary = is_primary

            log_dir = get_logdir_name(flags)
            graphpb_txt = str(tf.get_default_graph().as_graph_def())
            pathlib.Path(os.path.join(log_dir, 'eval')).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(log_dir, 'eval', 'graph.pbtxt'), 'w') as f:
                f.write(graphpb_txt)

    def eval(self, data_dir, num_cases_test, split='target_val'):
        data_set = Dataset(_load_mini_imagenet(data_dir=data_dir, split=split))

        num_batches = num_cases_test // self.batch_size
        num_correct = 0.0
        num_tot = 0.0
        loss_tot = 0.0
        final_alpha=[]
        for i in range(num_batches):
            num_classes, num_shots = self.flags.num_classes_test, self.flags.num_shots_test

            images_deploy, labels_deploy, images_task_encode, labels_task_encode_real, labels_task_encode  = \
                data_set.next_few_shot_batch_wordemb(deploy_batch_size=self.batch_size,
                                             num_classes_test=num_classes, num_shots=num_shots,
                                             num_tasks=1)


            feed_dict = {self.tensor_images_deploy: images_deploy.astype(dtype=np.float32),
                         self.tensor_labels_task_encode_real: labels_task_encode_real,
                         self.tensor_labels_deploy: labels_deploy,
                         self.tensor_labels_task_encode: labels_task_encode,
                         self.tensor_images_task_encode: images_task_encode.astype(dtype=np.float32)}
            [logits, loss, alpha] = self.sess.run([self.logits, self.loss, self.alpha], feed_dict)
            final_alpha.append(alpha)
            labels_deploy_pred = np.argmax(logits, axis=-1)

            num_matches = sum(labels_deploy_pred == labels_deploy)
            num_correct += num_matches
            num_tot += len(labels_deploy_pred)
            loss_tot += loss
        if split=='target_tst':
            log_dir = get_logdir_name(self.flags)
            pathlib.Path(os.path.join(log_dir, 'eval')).mkdir(parents=True, exist_ok=True)
            pkl.dump(final_alpha,open(os.path.join(log_dir, 'eval', 'lambdas.pkl'), "wb"))

        return num_correct / num_tot, loss_tot / num_batches


def get_few_shot_idxs(labels, classes, num_shots):
    train_idxs, test_idxs = [], []
    idxs = np.arange(len(labels))
    for cl in classes:
        class_idxs = idxs[labels == cl]
        class_idxs_train = np.random.choice(class_idxs, size=num_shots, replace=False)
        class_idxs_test = np.setxor1d(class_idxs, class_idxs_train)

        train_idxs.extend(class_idxs_train)
        test_idxs.extend(class_idxs_test)

    assert set(class_idxs_train).isdisjoint(test_idxs)

    return np.array(train_idxs), np.array(test_idxs)


def test(flags):
    test_dataset = _load_mini_imagenet(data_dir=flags.data_dir, split='target_val')

    # test_dataset = _load_mini_imagenet(data_dir=flags.data_dir, split='sources')
    images = test_dataset[0]
    labels = test_dataset[1]

    embedding_model = ModelLoader(flags.pretrained_model_dir, batch_size=100)
    embeddings = embedding_model.embed(images=test_dataset[0])
    embedding_model = None
    print("Accuracy test raw embedding: ", get_nearest_neighbour_acc(flags, embeddings, labels))


def get_agg_misclassification(logits_dict, labels_dict):
    summary_ops = []
    update_ops = {}
    for key, logits in logits_dict.items():
        accuracy, update = slim.metrics.streaming_accuracy(tf.argmax(logits, 1), labels_dict[key])

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            {'misclassification_' + key: (1.0 - accuracy, update)})

        for metric_name, metric_value in names_to_values.items():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        for update_name, update_op in names_to_updates.items():
            update_ops[update_name] = update_op
    return summary_ops, update_ops


def eval(flags, is_primary, fout):
    log_dir = get_logdir_name(flags)
    if is_primary:
        aux_prefix = ''
    else:
        aux_prefix = 'aux/'

    eval_writer = summary_writer(log_dir + '/eval')
    i = 0
    last_step = -1
    while i < flags.max_number_of_evaluations:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=flags.pretrained_model_dir)
        model_step = int(os.path.basename(latest_checkpoint or '0-0').split('-')[1])
        if last_step < model_step:
            results = {}
            model_train = ModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size,
                                is_primary=is_primary,split='train')
            acc_trn, loss_trn = model_train.eval(data_dir=flags.data_dir, num_cases_test=flags.num_cases_test,
                                                 split='sources')

            model_val = ModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size,
                                      is_primary=is_primary, split='val')
            acc_val, loss_val = model_val.eval(data_dir=flags.data_dir, num_cases_test=flags.num_cases_test,
                                               split='target_val')

            model_test = ModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size,
                                    is_primary=is_primary, split='test')
            acc_tst, loss_tst = model_test.eval(data_dir=flags.data_dir, num_cases_test=flags.num_cases_test,
                                           split='target_tst')

            results[aux_prefix + "accuracy_target_tst"] = acc_tst
            results[aux_prefix + "accuracy_target_val"] = acc_val
            results[aux_prefix + "accuracy_sources"] = acc_trn

            results[aux_prefix + "loss_target_tst"] = loss_tst
            results[aux_prefix + "loss_target_val"] = loss_val
            results[aux_prefix + "loss_sources"] = loss_trn

            last_step = model_train.step
            eval_writer(model_train.step, **results)
            logging.info("accuracy_%s: %.3g, accuracy_%s: %.3g, accuracy_%s: %.3g, loss_%s: %.3g, loss_%s: %.3g, loss_%s: %.3g."
                         % (
                         aux_prefix + "target_tst", acc_tst, aux_prefix + "target_val", acc_val, aux_prefix + "sources",
                         acc_trn, aux_prefix + "target_tst", loss_tst, aux_prefix + "target_val", loss_val, aux_prefix + "sources",
                         loss_trn))
            fout.write("accuracy_test: "+str(acc_tst)+" accuracy_val: "+str(acc_val)+" accuracy_test: "+str(acc_trn))
        if flags.eval_interval_secs > 0:
            time.sleep(flags.eval_interval_secs)
        i = i + 1





def image_augment(images):
    """

    :param images:
    :return:
    """
    pad_percent = 0.125
    flip_proba = 0.5
    image_size = images.shape[1]
    pad_size = int(pad_percent * image_size)
    max_crop = 2 * pad_size

    images_aug = np.pad(images, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    output = []
    for image in images_aug:
        if np.random.rand() < flip_proba:
            image = np.flip(image, axis=1)
        crop_val = np.random.randint(0, max_crop)
        image = image[crop_val:crop_val + image_size, crop_val:crop_val + image_size, :]
        output.append(image)
    return np.asarray(output)


def main(argv=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print(os.getcwd())

    default_params = get_arguments()
    log_dir = get_logdir_name(flags=default_params)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    # This makes sure that we can store a json and recove a namespace back
    flags = Namespace(load_and_save_params(vars(default_params), log_dir))

    if flags.mode == 'train':
        train(flags=flags)
    elif flags.mode == 'eval':
        eval(flags=flags, is_primary=True)
    elif flags.mode == 'test':
        test(flags=flags)


if __name__ == '__main__':
    tf.app.run()