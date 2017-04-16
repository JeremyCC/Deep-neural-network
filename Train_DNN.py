import tensorflow as tf
import model.DNN as DNN
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

######### Parameters ##########
save_dir='DNN'
num_layers=2
num_neurons=[100,80]
input_size=784
output_size=10
Reg=None #Regularization: L1,L2,None
alpha=0.1

dropout=0.5#(0,1.0]  1.0 mean without dropout
iteration=2000
step=50
fig,ax=plt.subplots(num_layers+1,2)
#fig.suptitle('Weights/Bias Distribution')
loss_all=np.zeros([int(iteration/step),2])
fig.tight_layout()
fig2=plt.figure()
fig2.suptitle('Learning curve')
learning_curve=fig2.add_subplot(111)
##############################

mymodel= DNN.model(num_layers,num_neurons,input_size,output_size,Reg,alpha)

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=10)
    writer = tf.summary.FileWriter(save_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        #batch_xs, batch_ys = mnist.train.next_batch(200)
        now=i%5
        batch_xs=mnist.train.images[i*200:(i+1)*200,:]
        batch_ys = mnist.train.labels[i * 200:(i + 1) * 200, :]
        _,accu,loss,merge=sess.run([mymodel.optimize,mymodel.accuracy,mymodel.cross_entropy,mymodel.sum_train],feed_dict={mymodel.x_in:batch_xs, mymodel.ys:batch_ys,mymodel.drop:dropout})
        writer.add_summary(merge,i)

        if i%step==0 :
            print("Iteration:{0}  Acu:{1}".format(i,accu))
            accu2,loss2,merge=sess.run([mymodel.accuracy,mymodel.cross_entropy,mymodel.sum_test],feed_dict={mymodel.x_in:mnist.test.images,mymodel.ys:mnist.test.labels,mymodel.drop:1.0})
            writer.add_summary(merge, i)
            loss_all[int(i/step),:]=[1-accu,1-accu2]

            print("Iteration:{0}  Accuracy:{1}".format(i,accu2))
            # #print('Save model to:saver/my-model-{}'.format(iteration))
            #saver.save(sess,save_dir+ '/my-model', global_step=iteration)


    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error rate', fontsize=16)
    learning_curve.plot(np.linspace(0,iteration,int(iteration/step),dtype=np.int32), loss_all[:,0],label='Training_errorrate')

    learning_curve.plot(np.linspace(0, iteration, int(iteration / step), dtype=np.int32), loss_all[:,1],label='Testing_errorrate')

    learning_curve.legend(loc='upper right')
    vars=tf.trainable_variables()
    all=[]
    pltall=[]
    for t in vars:
        S=sess.run(t)
        pltall.append(S)
        all.append(tf.summary.histogram(t.name+'_distribution',t))


    final=sess.run(tf.summary.merge(all))
    writer.add_summary(final, 0)

    for i in range(num_layers+1):
        for j in range(2):
            if j == 0:
                ax[i, j].set_title('hidden' + str(i) + '-Weights')

            else:
                ax[i, j].set_title('hidden' + str(i) + '-Bias')
            K=pltall[i * num_layers + j]
            K=np.reshape(K,[K.size])
            #D=np.histogram(K,bins=[0,0.01])
            ax[i, j].hist(K,bins=50)

    fig.savefig(save_dir+'/Distribution')
    fig2.savefig(save_dir+'/Learn')
    plt.show()