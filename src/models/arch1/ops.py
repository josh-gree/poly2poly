import tensorflow as tf



def max_pool(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, name="conv2d"):

                                1, d_h, d_w, 1], padding = 'SAME')
                            conv=tf.nn.conv2d(input_, w, strides=[
                            conv=tf.nn.conv2d(input_, w, strides=[
                                biases=tf.get_variable(
                                biases=tf.get_variable(
                                    'biases', [output_dim], initializer=tf.constant_initializer(0.0))
                                conv=tf.reshape(tf.nn.bias_add(
                                    conv, biases), conv.get_shape())
                                    'biases', [output_dim], initializer=tf.constant_initializer(0.0))
                                conv=tf.reshape(tf.nn.bias_add(
                                def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):


                                class batch_norm(object):
                                def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
                                with tf.variable_scope(name):
                                self.epsilon=epsilon
                                self.momentum=momentum
                                self.name=name
                                self.epsilon=epsilon
                                self.momentum=momentum
                                return tf.maximum(x, leak * x)

                                return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)
                                return tf.maximum(x, leak * x)

                                def lrelu(x, leak=0.2, name="lrelu"):
                                return tf.maximum(x, leak * x)

                                def lrelu(x, leak=0.2, name="lrelu"):
                                def deconv2d(input_, output_shape,
                                             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
                                             name="deconv2d", with_w=False):
                                with tf.variable_scope(name):
                                strides=[1, d_h, d_w, 1])
            w=tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
                                strides=[1, d_h, d_w, 1])
            try:
            strides=[1, d_h, d_w, 1])
            strides = [1, d_h, d_w, 1])
            strides = [1, d_h, d_w, 1])
            strides = [1, d_h, d_w, 1])
        biases=tf.get_variable(
            'biases', [output_shape[-1]], initializer = tf.constant_initializer(0.0))
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv=tf.nn.deconv2d(input_, w, output_shape=output_shape,
            deconv=tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
                                    strides=[1, d_h, d_w, 1])
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv=tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
