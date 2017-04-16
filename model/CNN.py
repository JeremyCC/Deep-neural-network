import tensorflow as tf

class model(object):
    def __init__(self,c_layers,f_layers,neuron,stride,kernel,input,output,featuremap):
        self.c_layers=c_layers
        self.f_layers=f_layers
        self.stride=stride#list of stride, number should be the same as c_layer
        self.kernel=kernel#list of [kernel_height, kernel_width, in_channels, out_channels]
        self.input_size=input
        self.num_neuron=neuron
        self.num_output=output
        self.last_featuremapsize=featuremap
        self.train_summary_list = []
        self.test_summary_list = []


        with tf.name_scope('Input'):
            self.xs=tf.placeholder(dtype=tf.float32,shape=self.input_size,name='x_input')
            self.ys_one = tf.placeholder(dtype=tf.float32,shape=[None,output] , name='target')
            self.drop = tf.placeholder(dtype=tf.float32, name='dropout_ratio')
            self.train_summary_list.append(tf.summary.image('input_image',self.xs,max_outputs=1))

        self.last=self.xs
        for i in range(c_layers):
            with tf.variable_scope('Convolution_layer'+str(i+1)):
                self.last=self.add_convolution_layer(self.last,self.kernel[i],self.stride[i])

        with tf.name_scope('Flatten'):
            self.last,shape=self.add_flat(self.last)
            self.num_neuron.insert(0,shape)

        for i in range(f_layers):
            with tf.variable_scope('Fully_connect_layer'+str(i+1)):
                self.last=self.add_fully(self.num_neuron[i],self.num_neuron[i+1],self.last)


        with tf.variable_scope('Output_layer'):
            self.output=self.add_output(self.num_neuron[-1],self.num_output,self.last)

        with tf.name_scope('Loss')    :
            self.cross_entropy = self.loss()

        with tf.name_scope('Optimizer'):
           self.optimize = tf.train.AdamOptimizer().minimize(self.cross_entropy)
        with tf.name_scope('Testing'):
            self.accuracy=self.testing()

        self.train_summary_list.append(tf.summary.scalar( 'Loss',self.cross_entropy))
        self.sum_train = tf.summary.merge(self.train_summary_list)
        self.test_summary_list.append(tf.summary.scalar( 'Loss_test',self.cross_entropy))
        self.test_summary_list.append(tf.summary.scalar('Accuracy', self.accuracy))
        self.sum_test = tf.summary.merge(self.test_summary_list)

    def add_convolution_layer(self,input,kernel_size,stride):
        kernel= self.weight_kernel(kernel_size)
        bias=self.bias_kernel(kernel_size[3])
        self.train_summary_list.append(tf.summary.histogram('weight_iter', kernel))
        self.train_summary_list.append(tf.summary.histogram('bias_iter', bias))

        conv=tf.nn.conv2d(input,kernel,stride,'SAME')
        hidden=tf.nn.relu(conv+bias)
        with tf.name_scope('Slice_observe_featuremap'):
           shape=tf.shape(hidden)
           observe=tf.slice(hidden,[0,0,0,0],[1,shape[1],shape[2],shape[3]])#only cut the first pic's feature map
           observe=tf.transpose(observe,[3,1,2,0])
           self.train_summary_list.append(tf.summary.image('feature_map', observe, max_outputs=15))
        hidden_pool=tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],'VALID',name='pooling')#max pooling of 2*2 stride2
        return hidden_pool

    def add_flat(self,cnn_in):
        shape=self.kernel[-1][-1] * self.last_featuremapsize ** 2
        data_re = tf.reshape(cnn_in,[-1,shape] )
        return data_re,shape

    def add_fully(self, size_in, size_out, data_in):
        W = self.weight_NN(size_in, size_out)
        B = self.bias_NN(size_out)
        self.train_summary_list.append(tf.summary.histogram('weight_iter', W))
        self.train_summary_list.append(tf.summary.histogram('bias_iter', B))
        data_drop = tf.nn.dropout(data_in, self.drop)
        result = tf.nn.relu(tf.matmul(data_drop, W) + B)
        return result

    def add_output(self,size_in, size_out, data_in):
        W = self.weight_NN(size_in,size_out)
        B = self.bias_NN(size_out)
        logits=tf.matmul(tf.nn.dropout(data_in,self.drop),W)+B
        self.train_summary_list.append(tf.summary.histogram('weight_iter', W))
        self.train_summary_list.append(tf.summary.histogram('bias_iter', B))
        K = tf.reduce_max(logits, 1)
        K = tf.reshape(K, [-1, 1])
        P = tf.tile(K, [1, 10])
        self.logits = logits - P
        result = tf.nn.softmax(self.logits)
        return result

    def loss(self):
        crossentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.ys_one,logits=self.logits)
        crossentropy=tf.reduce_mean(crossentropy)
        return crossentropy

    def testing(self):
        correct=tf.equal(tf.argmax(self.output,1),tf.argmax(self.ys_one,1))
        result=tf.reduce_mean(tf.cast(correct,tf.float32))
        return result

    def weight_kernel(self,size):#size [kernel_height, kernel_width, in_channels, out_channels]
         initial=tf.truncated_normal(size,stddev=0.01)
         return tf.Variable(initial,name='kernel')

    def bias_kernel(self,out_channel):
         initial=tf.constant(0.0,shape=[out_channel])
         return tf.Variable(initial,name='conv_bias')

    def weight_NN(self,w_in,w_out):
        initial=tf.truncated_normal([w_in,w_out],stddev=0.01)
        return tf.Variable(initial,name='weight')


    def bias_NN(self,out):
        initial=tf.constant(0.0,shape=[out])
        return tf.Variable(initial,name='bias')

