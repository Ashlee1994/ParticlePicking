import os,time
import tensorflow as tf
import numpy as np

class Vgg19:
    def __init__(self,args, decay_step = 500):
        self.args = args
        self.variables_vgg19()
        self.X = tf.placeholder(tf.float32, shape = [self.args.num_gpus, self.args.batch_size, self.args.boxsize, self.args.boxsize, 1])
        self.Y = tf.placeholder(tf.float32, shape = [self.args.num_gpus, self.args.batch_size,1])
        if self.args.is_training:
            self.lr = tf.constant(self.args.learning_rate, tf.float32)
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        # self.decay_step = tf.placeholder(tf.float32, shape = [1])
        self.decay_step = decay_step
        self._reuse_weights = True
        self.build_model()
        if self.args.is_training:
            self.build_train_op(args)

    def variables_vgg19(self):
        self.data_dict = {
            # filter = [kernelsize, kernelsize, last_layser_feature_map_num, feature_map_num]
            # biases = [feature_map_num]
            'conv1_1': {'filter': [3,3,1,  64],  'biases': [64]},
            'conv1_2': {'filter': [3,3,64, 64],  'biases': [64]},
            'conv2_1': {'filter': [3,3,64, 128], 'biases':[128]},
            'conv2_2': {'filter': [3,3,128,128], 'biases':[128]},
            'conv3_1': {'filter': [3,3,128,256], 'biases':[256]},
            'conv3_2': {'filter': [3,3,256,256], 'biases':[256]},
            'conv3_3': {'filter': [3,3,256,256], 'biases':[256]},
            'conv3_4': {'filter': [3,3,256,256], 'biases':[256]},
            'conv4_1': {'filter': [3,3,256,512], 'biases':[512]},
            'conv4_2': {'filter': [3,3,512,512], 'biases':[512]},
            'conv4_3': {'filter': [3,3,512,512], 'biases':[512]},
            'conv4_4': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_1': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_2': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_3': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_4': {'filter': [3,3,512,512], 'biases':[512]},
            'fc6': {'filter': [], 'biases':[4096]},    # 4096,
            'fc7': {'filter': [4096, 4096], 'biases':[4096]},    # 4096,
            'fc8': {'filter': [4096,    1], 'biases':[1]}   # 1
        }
        

    def VGG19_model(self, images, labels ):
        conv1_1 = self.conv_layer(images,       "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2,       'pool1')
        print("shape of conv1_1: ", conv1_1.shape)
        print("shape of conv1_2: ", conv1_2.shape)
        print("shape of pool1:   ", pool1.shape)

        conv2_1 = self.conv_layer(pool1,   "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2,       'pool2')
        print("shape of conv2_1: ", conv2_1.shape)
        print("shape of conv2_2: ", conv2_2.shape)
        print("shape of pool2:   ", pool2.shape)

        conv3_1 = self.conv_layer(pool2,   "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4,       'pool3')
        print("shape of conv3_1: ", conv3_1.shape)
        print("shape of conv3_2: ", conv3_2.shape)
        print("shape of conv3_3: ", conv3_3.shape)
        print("shape of conv3_4: ", conv3_4.shape)
        print("shape of pool3:   ", pool3.shape)

        conv4_1 = self.conv_layer(pool3,   "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4,       'pool4')
        print("shape of conv4_1: ", conv4_1.shape)
        print("shape of conv4_2: ", conv4_2.shape)
        print("shape of conv4_3: ", conv4_3.shape)
        print("shape of conv4_4: ", conv4_4.shape)
        print("shape of pool4:   ", pool4.shape)

        conv5_1 = self.conv_layer(pool4,   "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        pool5 = self.max_pool(conv5_4,       'pool5')
        print("shape of conv5_1: ", conv5_1.shape)
        print("shape of conv5_2: ", conv5_2.shape)
        print("shape of conv5_3: ", conv5_3.shape)
        print("shape of conv5_4: ", conv5_4.shape)
        print("shape of pool5:   ", pool5.shape)


        fc6 = self.fc_layer(pool5, "fc6")
        relu6 = tf.nn.relu(fc6)

        if self.args.is_training and self.args.dropout:
            relu6 =  tf.nn.dropout(relu6, self.args.dropout_rate)

        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        if self.args.is_training and self.args.dropout:
            relu7 =  tf.nn.dropout(relu7, self.args.dropout_rate)

        fc8 = self.fc_layer(relu7, "fc8")

        print("shape of fc6:     ", fc6.shape)
        print("shape of fc7:     ", fc7.shape)
        print("shape of fc8:     ", fc8.shape)
    
        # calculate the accuracy in the training set
        logits = fc8
        pred = tf.nn.sigmoid(logits, name="prob")
        if not self.args.is_training:
            return pred
        one = tf.ones_like(pred)
        zero = tf.zeros_like(pred)
        correct = tf.where(pred < 0.5 , x=zero, y = one)
        # print("tf.DataType:correct ", tf.type(correct))
        # print("tf.DataType:labels  ", tf.type(tf.cast(labels,tf.bool)))
        print("shape of pred:     ", pred.shape)
        print("shape of correct:     ", correct.shape)
        print("shape of labels:     ", labels.shape)
        corr = tf.where(tf.equal(correct,labels), one, zero)
        acc = tf.reduce_mean(corr)

        if not self.args.regularization:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels = labels))
        else:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels = labels)) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc6')) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc7')) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc8'))
                  
        # self.lr = tf.maximum(1e-12,tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_step, self.args.decay_rate, staircase=True))
        # self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.global_step)
        return logits, pred, loss, acc

    def build_model(self):
        # Split images and labels into (num_gpus) groups
        # images = tf.split(self._images, num_or_size_splits=self.num_gpus, axis=0)
        # labels = tf.split(self._labels, num_or_size_splits=self.num_gpus, axis=0)

        # Build towers for each GPU
        self._logits_list = []
        self._preds_list = []
        self._loss_list = []
        self._acc_list = []

        gpu_device = self.args.gpu_device.split(',')
        gpu_device = list(map(int,gpu_device))
        for i in range(self.args.num_gpus):
            # print("i: ", i)
            # print("gpu_device[i]: ", gpu_device[i])
            with tf.device('/GPU:%d' % gpu_device[i]), tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
                with tf.name_scope('tower_%d' % i) as scope:
                    # print('Build a tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()
                    
                    if self.args.is_training:
                        logits, preds, loss, acc = self.VGG19_model(self.X[i], self.Y[i])
                        self._logits_list.append(logits)
                        self._loss_list.append(loss)
                        self._acc_list.append(acc)
                    else:
                        preds = self.VGG19_model(self.X[i],  tf.constant(np.ones([self.args.batch_size]), dtype=tf.float32))
                    self._preds_list.append(preds)

        # Merge losses, accuracies of all GPUs
        with tf.device('/CPU:0'):
            self.preds = tf.concat(self._preds_list, axis=0, name="predictions")
            if self.args.is_training:
                self.logits = tf.concat(self._logits_list, axis=0, name="logits")
                self.loss = tf.reduce_mean(self._loss_list, name="cross_entropy")
                # tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy", self.loss)
                self.acc = tf.reduce_mean(self._acc_list, name="accuracy")
                # tf.summary.scalar((self._name+"/" if self._name else "") + "accuracy", self.acc)

    def build_train_op(self, args):
        self.lr = tf.maximum(1e-8,tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.args.decay_rate, staircase=True))

        if self.args.optimizer == "mom":
            opt = tf.train.MomentumOptimizer(self.lr, self.momentum)
        elif self.args.optimizer == "adam":
            opt = tf.train.AdamOptimizer(self.lr)
        elif self.args.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.lr)
        self._grads_and_vars_list = []

        # Computer gradients for each GPU
        self._logits_list = []
        self._preds_list = []
        self._loss_list = []
        self._acc_list = []

        gpu_device = self.args.gpu_device.split(',')
        gpu_device = list(map(int,gpu_device))
        for i in range(self.args.num_gpus):
            with tf.device('/GPU:%d' % gpu_device[i]), tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
                with tf.name_scope('tower_%d' % i) as scope:
                    # print('Compute gradients of tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # Add l2 loss
                    # costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                    # l2_loss = tf.multiply(self.args.reg_rate, tf.add_n(costs))
                    # total_loss = self._loss_list[i] + l2_loss
                    total_loss = self.loss

                    # Compute gradients of total loss
                    grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())

                    # Append gradients and vars
                    self._grads_and_vars_list.append(grads_and_vars)

        # Merge gradients
        print('Average gradients')
        with tf.device('/CPU:0'):
            grads_and_vars = self._average_gradients(self._grads_and_vars_list)

            # Finetuning
            if self.args.finetune:
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if "unit3" in var.op.name or \
                        "unit_last" in var.op.name or \
                        "/q" in var.op.name or \
                        "logits" in var.op.name:
                        print('\tScale up learning rate of % s by 10.0' % var.op.name)
                        grad = 10.0 * grad
                        grads_and_vars[idx] = (grad,var)

            # Apply gradient
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)

            # Batch normalization moving average update
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # self.train_op = tf.group(*(update_ops+[apply_grad_op]))
            self.train_op = apply_grad_op

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            if name == "fc6":
                self.data_dict['fc6']['filter'] = [dim, 4096]
            
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.Variable(tf.contrib.layers.xavier_initializer()(self.data_dict[name]['filter']))

    def get_bias(self, name):
        return tf.Variable(tf.zeros(self.data_dict[name]['biases']), name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(tf.contrib.layers.xavier_initializer()(self.data_dict[name]['filter']))
