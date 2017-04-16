import tensorflow as tf

class model(object):
    def __init__(self,layers,neuron,input,output,regularization,alpha):
        self.num_layer=layers
        self.num_neuron=[]

        self.num_neuron.append(input)
        for n in neuron:
            self.num_neuron.append(n)

        self.num_output=output
        self.num_input=input
        self.W=[]
        self.alpha=alpha
        self.train_summary_list=[]
        self.test_summary_list=[]
        with tf.name_scope('X_input'):
            self.x_in=tf.placeholder(dtype=tf.float32,shape=[None,input],name='data')
            self.ys=tf.placeholder(dtype=tf.float32,shape=[None,10],name='target')
            self.drop=tf.placeholder(dtype=tf.float32,name='dropout_ratio')

        self.last=self.x_in
        for i in range(layers):#construct layer iteratively
            with tf.variable_scope('Hidden_layer'+str(i)):
                self.last=self.add_hidden(self.num_neuron[i],neuron[i],self.last,i>0)

        with tf.variable_scope('Output_layer'):
            self.output=self.softmax()

        with tf.name_scope('Loss'):
            self.cross_entropy=self.loss(regularization)
        with tf.name_scope('Optimizer'):
            self.optimize = tf.train.AdamOptimizer().minimize(self.cross_entropy)
        with tf.name_scope('Testing'):
            self.accuracy=self.testing()

        self.train_summary_list.append(tf.summary.scalar( 'Loss',self.cross_entropy))
        self.test_summary_list.append(tf.summary.scalar('Accuracy',self.accuracy))

        self.sum_train=tf.summary.merge(self.train_summary_list)
        self.sum_test=tf.summary.merge(self.test_summary_list)

    def add_hidden(self,size_in,size_out,data_in,hiddenlayer=True):
        W=self.weighting(size_in,size_out)
        B=self.bias(size_out)
        self.W.append(W)
        self.train_summary_list.append(tf.summary.histogram(name='weight_flow', values=W))
        self.train_summary_list.append(tf.summary.histogram(name='bias_flow', values=B))
        if hiddenlayer:
          data_in=tf.nn.dropout(data_in,self.drop)

        result=(tf.matmul(data_in,W)+B)
        return result


    def softmax(self):
        W = self.weighting(self.num_neuron[-1], self.num_output)
        B = self.bias(self.num_output)
        logits=tf.matmul(tf.nn.dropout(self.last,self.drop),W)+B
        self.train_summary_list.append(tf.summary.histogram(name='weight_flow',values=W))
        self.train_summary_list.append(tf.summary.histogram(name='bias_flow', values=B))
        return tf.nn.softmax(logits,name='softmax')

    def loss(self,regularization):
        cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.ys*tf.log(self.output),1),name='MSE')
        cross_entropy=self.Reg(regularization,cross_entropy)
        return cross_entropy


    def Reg(self,regularization,cross_entropy):
        if regularization=='L1':
            print("L1")
            for W in self.W:
               cross_entropy +=self.alpha*tf.norm(W,1)
               return cross_entropy

        elif regularization=='L2':
            for W in self.W:
               cross_entropy+=self.alpha*tf.norm(W,2)
               return cross_entropy
        else:
               return cross_entropy



    def testing(self):
        correct=tf.equal(tf.argmax(self.output,1),tf.argmax(self.ys,1))
        result=tf.reduce_mean(tf.cast(correct,tf.float32))
        return result

    def weighting(self,w_in,w_out):
        W=tf.Variable(tf.truncated_normal([w_in,w_out],stddev=0.1),name='weight')
        return W


    def bias(self,out):
        B=tf.Variable(tf.constant(0,dtype=tf.float32,shape=[out]),name='bias')
        return B
