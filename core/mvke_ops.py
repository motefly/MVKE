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

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
      Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
      Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'): 
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
    with tf.device('/cpu:0'): 
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
    context = impl_combiner_impl.Context();
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
    return feature_output

def vcg_layer(tag_emb, vc_emb, vce_outputs):
    # tag_emb: [bz, emb_size] # vc_emb: [vc_num, emb_size] # vce_outputs: [bz, vc_num, emb_size]
    with tf.variable_scope("vcg", reuse=tf.AUTO_REUSE):
        vc_num = vc_emb.shape[0]
        emb_size = vc_emb.shape[1]

        # vc_emb = tf.layers.dense(vc_emb, emb_size, name="vc_emb_proj", 
        #     kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        vc_emb_trans = tf.transpose(vc_emb, [1,0]) # [emb_size, vc_num]
        dot_out = tf.matmul(tag_emb, vc_emb_trans) # [bz, vc_num]
        weights = tf.reshape(tf.nn.softmax(dot_out, 1), [-1, 1, vc_num]) # [bz, 1, vc_num]

        layer_output = tf.reshape(tf.matmul(weights, vce_outputs), [-1, emb_size]) # [bz, emb_size]

    return layer_output, tf.reshape(weights, [-1, vc_num]) # [bz, emb_size]

def mvke_bottom_layer(vc_emb, shared_field_embs):
    # vc_emb: [vc_num, emb_size] # shared_field_embs: [bz, feature_num, emb_size] 
    with tf.variable_scope("mvke_bottom", reuse=tf.AUTO_REUSE):
        vc_num = vc_emb.shape[0]
        emb_size = vc_emb.shape[1]
        feature_num = shared_field_embs.shape[1]

        query = tf.layers.dense(vc_emb, emb_size, name="query_proj", 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        reshaped_shared_field_embs = tf.reshape(shared_field_embs, [-1, emb_size])

        key = tf.layers.dense(reshaped_shared_field_embs, emb_size, name="key_proj", 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        value = tf.layers.dense(reshaped_shared_field_embs, emb_size, name="value_proj", 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)) 

        query_trans = tf.transpose(query, [1,0]) # [emb_size, vc_num]
        dot_out = tf.reshape(tf.matmul(key, query_trans) / math.sqrt(1.0*int(emb_size)), [-1, feature_num, vc_num]) # [bz, feature_num, vc_num]
        weights = tf.reshape(tf.nn.softmax(dot_out, 1), [-1, feature_num, vc_num, 1]) # [bz, feature_num, vc_num, 1]

        value = tf.reshape(value, [-1, feature_num, 1, emb_size]) # [-1, feature_num, 1, emb_size]
        value_expand = tf.tile(value, [1, 1, vc_num, 1]) # [bz, feature_num, vc_num, emb_size]
        layer_output = weights * value_expand

        layer_output = tf.contrib.layers.layer_norm(
            layer_output + tf.tile(tf.reshape(shared_field_embs, [-1, feature_num, 1, emb_size]), [1, 1, vc_num, 1]), 
            begin_norm_axis=-1, begin_params_axis=-1)

    return layer_output # [bz, feature_num, vc_num, emb_size]

def _build_mvke_layers(fields_config, field_name, features, feature_output, is_training, 
    tag_key, shared_model=False, vce_num=10, obj_config={'pctr': [0,5], 'pcvr': [2,9]}):

    if not field_name in fields_config:
        tf_logging.fatal("field_name: %s can not find config" % field_name)
    field_config = fields_config[field_name]

    layer_output, layout = process_impl_combiner(
        feature_output, field_config.combiner, is_training)
    feature_num = layout[1]
    emb_size = layout[2]
    with tf.variable_scope("mvke_"+field_name, reuse=tf.AUTO_REUSE):
        # [vce_num, emb_size]
        vc = _variable_on_cpu(
            "virtual_cluster", [vce_num , emb_size],
            tf.random_normal_initializer(stddev=0.01))
        mvke_bottom_out = mvke_bottom_layer(vc, layer_output) # [bz, feature_num, vc_num, emb_size]

        # share the same model in mvke
        if shared_model:
            layer_output = tf.reshape(tf.transpose(mvke_bottom_out, [0, 2, 1, 3]), [-1, feature_num, emb_size]) # [bz*vce_num, feature_num, emb_size]
            for i, layer in enumerate(field_config.field_layer.layer):
                with tf.variable_scope("field_layer_%d" % i):
                    layout, layer_output = process_impl_layer(
                        layer_output, layout, field_name, layer,
                        is_training, features) # [bz*vce_num, emb_size]
        else:
            layer_output_all = []
            for vce_idx in range(vce_num):
                layer_output = tf.reshape(tf.gather(mvke_bottom_out, vce_idx, axis=2), [-1, feature_num, emb_size]) # [bz, feature_num, emb_size]
                in_layout = layout
                for i, layer in enumerate(field_config.field_layer.layer):
                    with tf.variable_scope("vce_{}_field_layer_{}".format(vce_idx, i)):
                        in_layout, layer_output = process_impl_layer(
                            layer_output, in_layout, field_name, layer,
                            is_training, features)
                layer_output_all.append(tf.expand_dims(layer_output, 1))
            layer_output = tf.stack(layer_output_all, 1) # [bz, vce_num, emb_size]

        layer_output = tf.reshape(layer_output, [-1, vce_num, emb_size])
        final_layer = {}
        vce_weights = {}
        vce_outputs = {}
        for key in tag_key:
            tag_emb = tag_key[key]
            select_begin, select_end = obj_config[key]
            select_index = [i for i in range(select_begin, select_end+1)]
            select_vc_emb = tf.gather(vc, select_index, axis=0) # to-check
            select_layer_output = tf.gather(layer_output, select_index, axis=1) # to-check
            layer_out, vce_weight = vcg_layer(tag_emb, select_vc_emb, select_layer_output) # [bz, emb_size]

            final_layer[key] = layer_out
            vce_weights[key] = vce_weight
            vce_outputs[key] = tf.reshape(select_layer_output, [-1, (select_end-select_begin+1)*emb_size])

    return final_layer, vce_weights, vce_outputs

def _build_mvke(impl_config, features, field_name, is_training, tag_key, shared_model, vce_num, obj_config):
    features_config, fields_config, _ = impl_config
    feature_inputs, local_feature_config = process_feature_input(
        features_config, features, field_name)
    feature_output = process_feature_layers(
        feature_inputs, features, local_feature_config, is_training)
    layer_output = _build_mvke_layers(
        fields_config, field_name, features, feature_output, 
        is_training, tag_key, shared_model, vce_num, obj_config)

    return layer_output

def build_mvke(impl_config, features, field_name, is_training, tag_key, shared_model, vce_num, obj_config):
    """Build mvke for user"""
    feature_config, _, _ = impl_config
    field_features = {}
    for feature_name in features.keys():
        if field_name in feature_config[feature_name].field:
            field_features[feature_name] = features[feature_name]

    return _build_mvke(impl_config, field_features, field_name, 
        is_training, tag_key, shared_model, vce_num, obj_config)


def mmoe_gate_layer(inputs, expert_num, expert_outputs, task_name):
    # inputs: [bz, feature_num, emb_size] expert_outputs: [bz, expert_num, out_size]
    feature_num = inputs.shape[1]
    emb_size = inputs.shape[2]
    out_size = expert_outputs.shape[2]
    if expert_num == 1:
        return tf.reshape(expert_outputs, [-1, out_size])
    with tf.variable_scope("mmoe_gates_" + task_name, reuse=tf.AUTO_REUSE):
        weights = tf.layers.dense(tf.reshape(inputs, [-1, feature_num * emb_size]), expert_num, name="input_gate_dense", 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)) # [bz, expert_num]
        weights = tf.reshape(tf.nn.softmax(weights, 1), [-1, 1, expert_num]) # [bz, 1, expert_num]

        layer_output = tf.reshape(tf.matmul(weights, expert_outputs), [-1, out_size]) # [bz, out_size]
    return layer_output

def _build_mmoe_layers(fields_config, field_name, features, feature_output, is_training, 
    expert_num=10, obj_config={'pctr': [0,0], 'pcvr': [1,1]}):

    if not field_name in fields_config:
        tf_logging.fatal("field_name: %s can not find config" % field_name)
    field_config = fields_config[field_name]

    layer_output, layout = process_impl_combiner(
        feature_output, field_config.combiner, is_training)
    
    with tf.variable_scope("mmoe_"+field_name, reuse=tf.AUTO_REUSE):
        # experts
        expert_output_all = []
        for vce_idx in range(expert_num):
            expert_output = layer_output
            expert_layout = layout
            for i, layer in enumerate(field_config.field_layer.layer):
                with tf.variable_scope("expert_{}_field_layer_{}".format(vce_idx, i)):
                    expert_layout, expert_output = process_impl_layer(
                        expert_output, expert_layout, field_name, layer,
                        is_training, features)
            expert_output_all.append(expert_output) # [bz, emb_size]
        expert_output = tf.stack(expert_output_all, 1) # [bz, vce_num, emb_size]

        # task gates
        final_layer = {}
        for key in obj_config:
            select_begin, select_end = obj_config[key]
            select_index = [i for i in range(select_begin, select_end+1)]
            select_expert_output = tf.gather(expert_output, select_index, axis=1) # to-check
            final_layer[key] = mmoe_gate_layer(layer_output, select_end-select_begin+1,
                select_expert_output, key) # [bz, emb_size]
    
    return final_layer

def _build_mmoe(impl_config, features, field_name, is_training, expert_num, obj_config):
    features_config, fields_config, _ = impl_config
    feature_inputs, local_feature_config = process_feature_input(
        features_config, features, field_name)
    feature_output = process_feature_layers(
        feature_inputs, features, local_feature_config, is_training)
    layer_output = _build_mmoe_layers(
        fields_config, field_name, features, feature_output, 
        is_training, expert_num, obj_config)
    
    return layer_output

# support both mmoe and hard-shared bottom architecture
def build_mmoe(impl_config, features, field_name, is_training, expert_num, obj_config):
    """Build mmoe module"""
    feature_config, _, _ = impl_config
    field_features = {}
    for feature_name in features.keys():
        if field_name in feature_config[feature_name].field:
            field_features[feature_name] = features[feature_name]

    return _build_mmoe(impl_config, features, field_name, is_training, expert_num, obj_config)
