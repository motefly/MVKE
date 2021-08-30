# -*- coding:utf-8 -*-

import imp
import math
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.platform import tf_logging
from core import dataset_ops
from core import embedding_ops
from core import impl_layer_impl
from core import impl_combiner_impl
from core.impl_config_pb2 import ImplConfig

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('impl_config_variable_name',
                           'impl_config_variable', '')

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
      Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
      Returns:
        Variable Tensor
    """
    with tf.device('/gpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer)
    return var

def _variable_on_cpu_var(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
      Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
      Returns:
        Variable Tensor
    """
    with tf.device('/gpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, validate_shape=False)
    return var


def create_impl_config_variable():
    impl_config_variable = tf.get_variable(
        FLAGS.impl_config_variable_name, [], dtype=tf.string, trainable=False,
        initializer=tf.constant_initializer(
            dataset_ops.GetImplConfigContent(), dtype=tf.string))
    return impl_config_variable


def add_embedding_layer(features, feature_key, feature_config, seed=None):
    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        if FLAGS.use_impl_hash:
            if seed is None:
                raise Exception("seed is None in add_embedding_layer impl")
            feature_key_hash_name = "%s/%s"%(feature_key, seed.decode())
        else:
            feature_key_hash_name = feature_key
        bucket_size = feature_config.embedding.vocabulary_size
        output_size = feature_config.embedding.embedding_size
        w = _variable_on_cpu(
            feature_key_hash_name, [bucket_size, output_size],
            tf.random_normal_initializer(stddev=0.01))
        is_padding = feature_config.is_multi_value

        feature_weight_name = "%s:weights" % (feature_config.feature_name)
        feature_weight = None
        if feature_weight_name in features:
            feature_weight = features[feature_weight_name]

        if FLAGS.use_impl_hash:
            return embedding_ops.embedding_impl(
                w, features[feature_key], feature_weight,
                padding=is_padding, seed=seed.decode())

        return embedding_ops.embedding(
            w, features[feature_key], feature_weight,
            padding=is_padding, name=feature_key)


def process_impl_combiner(feature_output, combiner, is_training):
    context = impl_combiner_impl.Context()
    context.feature_output = feature_output
    context.is_training = is_training
    for m in combiner.ListFields():
        tf_logging.info('impl_combiner, member:%s' %(str(m[0].name)))
        method = None
        try:
            with tf.variable_scope(m[0].name) :
                method = getattr(impl_combiner_impl, m[0].name)
                context.param = m[1]
                return method(context)
        except AttributeError as e:
            tf_logging.warning('impl_combiner method:%s error:%s' %
                (str(m[0].name), str(e)))
    return None


def process_impl_layer(input, layout, layer_name, layer, is_training, features=None):
    context = impl_layer_impl.Context()
    context.input = input
    context.layout = layout
    context.is_training = is_training
    context.features = features
    context.layer_name = layer_name
    for m in layer.ListFields():
        tf_logging.info('feature_name:%s, member:%s, layout:%s' %
            (layer_name, str(m[0].name), str(layout)))
        method = None
        try:
            with tf.variable_scope(m[0].name, reuse=tf.AUTO_REUSE):
                method = getattr(impl_layer_impl, m[0].name)
                context.param = m[1]
                return method(context)
        except AttributeError as e:
            tf_logging.warning('feature:%s, method:%s error:%s' %
                (layer_name, str(m[0].name), str(e)))
    return layout, input


def process_feature_input(features_config, features, field_name):
    feature_inputs = {}
    local_feature_config = {}
    for feature_key, feature_value in features.items():
        if feature_value is None:
            tf_logging.fatal("feature_value %s is None. Quit" % feature_key)

        feature_config = features_config[feature_key]
        if field_name not in feature_config.field:
            continue
        if feature_config.is_weight_feature:
            continue
        tf_logging.info('process feature: %s' % feature_key)
        if feature_config.type == ImplConfig.Feature.BYTES:
            tf_logging.fatal("bytes feature %s is not support" % feature_key)
        feature_name = feature_config.feature_name
        if feature_config.HasField('embedding'):
            feature_inputs.setdefault(feature_name, [])
            if FLAGS.use_impl_hash and feature_config.HasField('bucketing'):
                for seed in feature_config.bucketing.seeds:
                    feature_inputs[feature_name].append(
                        add_embedding_layer(features, feature_key, feature_config, seed=seed))
            else:
                feature_inputs[feature_name].append(
                    add_embedding_layer(features, feature_key, feature_config))
            local_feature_config[feature_name] = feature_config
        else:
            feature_inputs.setdefault(feature_name, [])
            feature_inputs[feature_name].append(features[feature_key])
            local_feature_config[feature_name] = feature_config
    return feature_inputs, local_feature_config

def process_feature_layers(feature_inputs, features,
                           local_feature_config, is_training, custom_scope=None):
    class FeatureOutputItem:
        pass
    feature_output = {}
    for feature_name, input_values in feature_inputs.items():
        feature_config = local_feature_config[feature_name]
        feature_dimension = None
        if not feature_config.is_multi_value:
            if feature_config.feature_dimension != 0:
                feature_dimension = feature_config.feature_dimension
            else:
                feature_dimension = 1
        embed_size = 1
        if feature_config.HasField('embedding'):
            embed_size = feature_config.embedding.embedding_size
        seed_num = len(input_values)
        # Construct feature tensor, Reshape to
        # [BatchSize, FeatureDimension, EmbeddingSize, SeedNum]
        feature_tensor = tf.stack(input_values, -1)
        batch_size = tf.shape(feature_tensor)[0]
        shape = [batch_size, -1, embed_size, seed_num]
        feature_tensor = tf.reshape(feature_tensor, shape)
        # Process Feature Layers
        layer_output = feature_tensor
        layout = [None, feature_dimension, embed_size, seed_num]
        if feature_config.HasField('feature_layer'):
            for i, layer in enumerate(feature_config.feature_layer.layer):
                scope_name = feature_config.feature_name
                if custom_scope is not None:
                    scope_name == custom_scope + "_" + scope_name
                with tf.variable_scope("%s/feature_layer_%d" %
                    (scope_name, i), reuse=tf.AUTO_REUSE):
                    layout, layer_output = process_impl_layer(
                        layer_output, layout, feature_config.feature_name,
                        layer, is_training, features)
        feature_output.setdefault(feature_name, FeatureOutputItem())
        feature_output[feature_name].output = layer_output
        # feature_output[feature_name].input = feature_tensor
        feature_output[feature_name].layout = layout
        feature_output[feature_name].name = feature_name
        tf_logging.info("feature_name: %s, layout: %s" %
            (feature_name, str(layout)))
    return feature_output;


def build_field_layer(fields_config, field_name, features,
                      feature_output, is_training, custom_scope=None):
    if not field_name in fields_config:
        tf_logging.fatal("field_name: %s can not find config" % field_name)
    field_config = fields_config[field_name]

    layer_output, layout = process_impl_combiner(
        feature_output, field_config.combiner, is_training)
    bug_output = layer_output
    
    scope_name = field_name
    if custom_scope is not None:
        scope_name = custom_scope + "_" + field_name

    with tf.variable_scope("field_%s" % scope_name):
        for i, layer in enumerate(field_config.field_layer.layer):
            with tf.variable_scope("field_layer_%d" % i):
                layout, layer_output = process_impl_layer(
                    layer_output, layout, field_name, layer,
                    is_training, features)

    return layer_output


def _build_field(impl_config, features, field_name, is_training):
    features_config, fields_config, _ = impl_config
    feature_inputs, local_feature_config = process_feature_input(
        features_config, features, field_name)
    feature_output = process_feature_layers(
        feature_inputs, features, local_feature_config, is_training)
    layer_output = build_field_layer(
        fields_config, field_name,
        features, feature_output, is_training)

    # vals = [(k, feature_output[k].input) for k in feature_output.keys()]
    # new_vals = []
    # for k, v in vals:
    #     bv = tf.reduce_sum(tf.cast(tf.is_nan(v), tf.int64))
    #     print(k, bv)
    #     new_vals.append((k, bv))
    return layer_output


def build_field(impl_config, features, is_training, field_name):
    """Build network of field_name"""
    feature_config, _, _ = impl_config
    field_features = {}
    for feature_name in features.keys():
        # beatlesyang
        # if feature_name == "user_id_real":
        #     continue
        if field_name in feature_config[feature_name].field:
            field_features[feature_name] = features[feature_name]

    return _build_field(impl_config, field_features, field_name, is_training)

def build_context_field(impl_config, features, is_training, field_name):
    """Build Context network of field_name"""
    feature_config, _, _ = impl_config
    field_features = {}
    for feature_name in features.keys():
        if field_name in feature_config[feature_name].field:
            field_features[feature_name] = features[feature_name]
            field_features[feature_name] = tf.expand_dims(field_features[feature_name], 0)

    return _build_field(impl_config, field_features, field_name, is_training)

def _build_mtl_shared_field(impl_config, features, field_name, is_training, task_names):
    features_config, fields_config, _ = impl_config
    feature_inputs, local_feature_config = process_feature_input(
        features_config, features, field_name)
    layer_outputs = {}
    for task in task_names:
        feature_output = process_feature_layers(
            feature_inputs, features, local_feature_config, is_training, custom_scope=task)
        layer_output = build_field_layer(
            fields_config, field_name,
            features, feature_output, is_training, custom_scope=task)
        layer_outputs[task] = layer_output
    return layer_outputs

def build_mtl_shared_field(impl_config, features, is_training, field_name, task_names):
    """Build network of field_name"""
    feature_config, _, _ = impl_config
    field_features = {}
    for feature_name in features.keys():
        if field_name in feature_config[feature_name].field:
            field_features[feature_name] = features[feature_name]

    return _build_mtl_shared_field(impl_config, field_features, field_name, is_training, task_names)

def _build_mtl_user_field(impl_config, features, field_name, is_training, tag_key, task_names):
    features_config, fields_config, _ = impl_config
    feature_inputs, local_feature_config = process_feature_input(
        features_config, features, field_name)
    layer_outputs = {}
    for task in task_names:
        feature_output = process_feature_layers(
            feature_inputs, features, local_feature_config, is_training, custom_scope=task)
        layer_output = build_user_field_layer(
            fields_config, field_name,
            features, feature_output, is_training, tag_key, custom_scope=task)
        layer_outputs[task] = layer_output
    return layer_outputs

def build_mtl_user_field(impl_config, features, is_training, field_name, tag_key, task_names):
    """Build network of field_name"""
    feature_config, _, _ = impl_config
    field_features = {}
    for feature_name in features.keys():
        if field_name in feature_config[feature_name].field:
            field_features[feature_name] = features[feature_name]

    return _build_mtl_user_field(impl_config, field_features, field_name, is_training, tag_key, task_names)
