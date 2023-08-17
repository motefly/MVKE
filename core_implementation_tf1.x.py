def _mvke_bottom_layer(self, vk_emb, shared_field_embs):
        # vk_emb: [vk_num, emb_size] # shared_field_embs: [bz, feature_num, emb_size] 
        with tf.variable_scope("mvke_bottom", reuse=tf.AUTO_REUSE):
            feature_num = shared_field_embs.shape[1]
            feature_emb_size = shared_field_embs.shape[2]

            query = tf.layers.dense(vk_emb, feature_emb_size, name="query_proj", 
                kernel_initializer=tf.random_normal_initializer(stddev=0.01))

            reshaped_shared_field_embs = tf.reshape(shared_field_embs, [-1, feature_emb_size])

            key = tf.layers.dense(reshaped_shared_field_embs, feature_emb_size, name="key_proj", 
                kernel_initializer=tf.random_normal_initializer(stddev=0.01))

            value = tf.layers.dense(reshaped_shared_field_embs, feature_emb_size, name="value_proj", 
                kernel_initializer=tf.random_normal_initializer(stddev=0.01)) 

            query_trans = tf.transpose(query, [1,0]) # [emb_size, vk_num]
            dot_out = tf.reshape(tf.matmul(key, query_trans) / math.sqrt(1.0*int(feature_emb_size)), [-1, feature_num, self.vk_num]) # [bz, feature_num, vk_num]
            weights = tf.reshape(tf.nn.softmax(dot_out, 1), [-1, feature_num, self.vk_num, 1]) # [bz, feature_num, vk_num, 1]

            value = tf.reshape(value, [-1, feature_num, 1, feature_emb_size]) # [-1, feature_num, 1, emb_size]
            value_expand = tf.tile(value, [1, 1, self.vk_num, 1]) # [bz, feature_num, vk_num, emb_size]
            layer_output = weights * value_expand

            gamma_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.3)
            beta_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.0001)

            layer_output = layers_lib.layer_normalization(
                tf.reshape(layer_output + tf.tile(tf.reshape(shared_field_embs, [-1, feature_num, 1, feature_emb_size]), [1, 1, self.vk_num, 1]),[-1, feature_num*self.vk_num*feature_emb_size]),
                beta_initializer=beta_init,
                gamma_initializer=gamma_init,
                name="LN")

        return tf.reshape(layer_output, [-1, feature_num, self.vk_num, feature_emb_size])# [bz, feature_num, vk_num, emb_size]

    def _vkg_attn_layer(self, ad_emb):#, vke_outputs):
        # ad_emb: [bz, ad_len, emb_size] # vk_emb: [vk_num, emb_size] # vke_outputs: [bz, vk_num, emb_size]
        with tf.variable_scope("vkg_attn", reuse=tf.AUTO_REUSE):
            # vk_emb = tf.layers.dense(vk_emb, emb_size, name="vk_emb_proj", 
            #     kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            batch_size = tf.shape(ad_emb)[0]
            ad_emb = tf.reshape(ad_emb, [-1, self.vk_emb_size]) # [bz*ad_len, emb_size]
            vk_emb_trans = tf.transpose(self.virtual_kernel, [1, 0]) # [emb_size, vk_num]
            dot_out = tf.matmul(ad_emb, vk_emb_trans) # [bz*ad_len, vk_num]

            weights = tf.nn.softmax(dot_out, 1) # [bz*ad_len, vk_num]
            # layer_output = tf.reshape(tf.matmul(weights, vke_outputs), [-1, self.vk_emb_size]) # [bz, emb_size]

        return tf.reshape(weights, [-1, self.vk_num]) # [bz, vk_num]
