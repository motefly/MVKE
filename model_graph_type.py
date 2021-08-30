#!/usr/bin/env python

"""Provide Some Model Graphs"""

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from core import mvke_ops, model_ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_graph_type', 'normal_model_graph', 'normal model graph')
tf.app.flags.DEFINE_float('scale_k', 1.0, '')
tf.app.flags.DEFINE_float('user_norm_coef', 0.0, '')
tf.app.flags.DEFINE_float('ad_norm_coef', 0.0, '')

def get_model_graph_func(graph_type):
    """Get Model Graph Function"""
    if graph_type in globals():
        return globals()[graph_type]
    tf_logging.fatal("model_name=%s is not valid" % graph_type)


def get_labels(features_config, features, multi_task_labels=False):
    """Get Label Vector From Data"""
    if multi_task_labels:
        labels = {}
        for feature_name, feature_config in features_config.items():
            if feature_config.is_label_feature:
                label = tf.cast(tf.reshape(features[feature_name], [-1, 1]), tf.float32)
                labels[feature_name] = label
    else:
        labels = None
        for feature_name, feature_config in features_config.items():
            if feature_config.is_label_feature:
                labels = features[feature_name]
                break
        labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float32)
    return labels

def get_by_name(features_config, features, name):
    ret = {}
    for feature_name, feature_config in features_config.items():
        if name == feature_config.field:
            ret[feature_name] = tf.cast(tf.reshape(features[feature_name], [-1, 1]), tf.float32)
    return ret

def get_field_by_name(features_config, features, name):                               
    ret = None                                                                    
    for feature_name, feature_config in features_config.items():                
        if name == feature_name:                                        
            return tf.cast(tf.reshape(features[feature_name], [-1, 1]), tf.float32)
    return ret                    

def get_field_logits(impl_config, features, field_name, is_training):
    """Build Field Logtis"""
    logits = model_ops.build_field(
        impl_config, features, is_training, field_name)
    logits = tf.reshape(logits, [-1, 1])
    return logits

#---------------------------By xxxx------MVKE--------------------------
def user_MVKE_and_tag_HS(impl_config, features, is_training):
    """Dot model: user equipped MVKE & tag equipped Hard-Shared"""
    tf_logging.info("build_graph:%s" % str(features))

    features_config, fields_config, _ = impl_config
    field_name_list = list(fields_config.keys())
    if len(field_name_list) != 2:
        tf_logging.fatal("field count should equal to 2")

    field_name0 = "user"
    field_name1 = "tag"

    raw_labels = get_labels(features_config, features, multi_task_labels=True)
    labels = {'pctr': raw_labels['litectr_label'], 'pcvr': raw_labels['cvr_label']}

    tag_outs = mvke_ops.build_mmoe(impl_config, features, field_name1, is_training, 
        expert_num=2,
        obj_config={'pctr':[0,0], 'pcvr':[1,1]}) # 2 tasks have no shared expert

    user_outs, tag_vke_weights, user_vke_outputs = mvke_ops.build_mvke(impl_config, features, field_name0, is_training, 
                                                                       tag_key=tag_outs,
                                                                       shared_model=False, #FLAGS.shared_model, 
                                                                       vke_num=5, 
                                                                       obj_config={'pctr':[0,2], 'pcvr':[1,4]})

    losses = {}
    scores = {}
    pred_out_tag = {} 
    for task in ['pctr', 'pcvr']:
        user_emb = user_outs[task]
        tag_emb = tag_outs[task]
        label = labels[task]
        
        dot = tf.reduce_sum(tf.multiply(user_emb, tag_emb), axis=1, keepdims=True)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dot)
        loss = tf.reduce_mean(cross_entropy)
        
        losses[task] = loss
        scores[task] = tf.sigmoid(dot)
        pred_out_tag[task] = tf.concat([tag_emb, tag_vke_weights[task]], axis=1)
    
    loss_all = losses['pctr'] + losses['pcvr'] # can try more param here
    
    return {
        "loss": loss_all,
        "scores": scores,
        "labels": labels,
        # for predict
        field_name0: user_vke_outputs,
        field_name1: pred_out_tag,
    }

#---------------------------By xxxx------MMoE--------------------------
def user_MMoE_and_tag_HS(impl_config, features, is_training):
    """Dot model: user equipped MMoE & tag equipped Hard-Shared"""
    tf_logging.info("build_graph:%s" % str(features))

    features_config, fields_config, _ = impl_config
    field_name_list = list(fields_config.keys())
    if len(field_name_list) != 2:
        tf_logging.fatal("field count should equal to 2")

    field_name0 = "user"
    field_name1 = "tag"

    raw_labels = get_labels(features_config, features, multi_task_labels=True)
    labels = {'pctr': raw_labels['litectr_label'], 'pcvr': raw_labels['cvr_label']}

    tag_outs = mvke_ops.build_mmoe(impl_config, features, field_name1, is_training,
        expert_num=2,
        obj_config={'pctr':[0,0], 'pcvr':[1,1]}) # two tasks have no shared expert

    user_outs = mvke_ops.build_mmoe(impl_config, features, field_name0, is_training,
                                    expert_num=10,
                                    obj_config={'pctr':[0,5], 'pcvr':[2,9]})

    losses = {}
    scores = {}
    for task in ['pctr', 'pcvr']:
        user_emb = user_outs[task]
        tag_emb = tag_outs[task]
        label = labels[task]

        dot = tf.reduce_sum(tf.multiply(user_emb, tag_emb), axis=1, keepdims=True)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dot)
        loss = tf.reduce_mean(cross_entropy)

        losses[task] = loss
        scores[task] = tf.sigmoid(dot)

    loss_all = losses['pctr'] + losses['pcvr']

    return {
        "loss": loss_all,
        "scores": scores,
        "labels": labels,
        # for predict
        field_name0: user_outs,
        field_name1: tag_outs,
    }

#---------------------------By xxxx------HS--------------------------
def user_HS_and_tag_HS(impl_config, features, is_training):
    """Dot model: user equipped HS & tag equipped Hard-Shared"""
    tf_logging.info("build_graph:%s" % str(features))

    features_config, fields_config, _ = impl_config
    field_name_list = list(fields_config.keys())
    if len(field_name_list) != 2:
        tf_logging.fatal("field count should equal to 2")

    field_name0 = "user"
    field_name1 = "tag"

    raw_labels = get_labels(features_config, features, multi_task_labels=True)
    labels = {'pctr': raw_labels['litectr_label'], 'pcvr': raw_labels['cvr_label']}

    tag_outs = mvke_ops.build_mmoe(impl_config, features, field_name1, is_training,
        expert_num=2,
        obj_config={'pctr':[0,0], 'pcvr':[1,1]}) # two tasks have no shared expert

    user_outs = mvke_ops.build_mmoe(impl_config, features, field_name0, is_training,
        expert_num=2,
        obj_config={'pctr':[0,0], 'pcvr':[1,1]})

    losses = {}
    scores = {}
    for task in ['pctr', 'pcvr']:
        user_emb = user_outs[task]
        tag_emb = tag_outs[task]
        label = labels[task]

        dot = tf.reduce_sum(tf.multiply(user_emb, tag_emb), axis=1, keepdims=True)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dot)
        loss = tf.reduce_mean(cross_entropy)

        losses[task] = loss
        scores[task] = tf.sigmoid(dot)

    loss_all = losses['pctr'] + losses['pcvr']

    return {
        "loss": loss_all,
        "scores": scores,
        "labels": labels,
        # for predict
        field_name0: user_outs,
        field_name1: tag_outs,
    }

def user_and_tag_single_pctr(impl_config, features, is_training):
    """Dot model: user & tag both equipped single model"""
    tf_logging.info("build_graph:%s" % str(features))
    
    task = 'pctr'
    
    features_config, fields_config, _ = impl_config
    field_name_list = list(fields_config.keys())
    if len(field_name_list) != 2:
        tf_logging.fatal("field count should equal to 2")

    field_name0 = "user"
    field_name1 = "tag"

    raw_labels = get_labels(features_config, features, multi_task_labels=True)
    labels = {'pctr': raw_labels['litectr_label'], 'pcvr': raw_labels['cvr_label']}

    tag_outs = mvke_ops.build_mmoe(impl_config, features, field_name1, is_training,
        expert_num=1,
        obj_config={task:[0,0]})

    user_outs = mvke_ops.build_mmoe(impl_config, features, field_name0, is_training,
        expert_num=1,
        obj_config={task:[0,0]})

    user_emb = user_outs[task]
    tag_emb = tag_outs[task]
    label = labels[task]

    dot = tf.reduce_sum(tf.multiply(user_emb, tag_emb), axis=1, keepdims=True)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dot)
    loss = tf.reduce_mean(cross_entropy)
    scores = {task: tf.sigmoid(dot)}

    return {
        "loss": loss,
        "scores": scores,
        "labels": labels,
        # for predict
        field_name0: user_outs,
        field_name1: tag_outs,
    }

def user_and_tag_single_pcvr(impl_config, features, is_training):
    """Dot model: user & tag both equipped single model"""
    tf_logging.info("build_graph:%s" % str(features))
    
    task = 'pcvr'
    
    features_config, fields_config, _ = impl_config
    field_name_list = list(fields_config.keys())
    if len(field_name_list) != 2:
        tf_logging.fatal("field count should equal to 2")

    field_name0 = "user"
    field_name1 = "tag"

    raw_labels = get_labels(features_config, features, multi_task_labels=True)
    labels = {'pctr': raw_labels['litectr_label'], 'pcvr': raw_labels['cvr_label']}

    tag_outs = mvke_ops.build_mmoe(impl_config, features, field_name1, is_training,
        expert_num=1,
        obj_config={task:[0,0]})

    user_outs = mvke_ops.build_mmoe(impl_config, features, field_name0, is_training,
        expert_num=1,
        obj_config={task:[0,0]})

    user_emb = user_outs[task]
    tag_emb = tag_outs[task]
    label = labels[task]

    dot = tf.reduce_sum(tf.multiply(user_emb, tag_emb), axis=1, keepdims=True)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dot)
    loss = tf.reduce_mean(cross_entropy)
    scores = {task: tf.sigmoid(dot)}

    return {
        "loss": loss,
        "scores": scores,
        "labels": labels,
        # for predict
        field_name0: user_outs,
        field_name1: tag_outs,
    }
