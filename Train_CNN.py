########################################
# Training a CNN network using cifar-10
# Make sure All training and testing
# data are put in directory "cifar10"
#
# JeremyCC,2017/4/16
########################################
import tensorflow as tf
import model.CNN as CNN
import numpy as np

######### Parameters ##########
save_dir='CNN'
num_conv=3
num_fully=2
num_neurons=[150,80]

stride=[]
num_data_batch=100#must be the factor of 10000 and 50000
total_batch=int(50000*2/num_data_batch)
for i in range(num_conv):
    stride.append([1,1,1,1])

kernel=[]#list of [kernel_height, kernel_width, in_channels, out_channels]
kernel.append(np.array([3,3,3,32]))
kernel.append(np.array([5,5,32,64]))
kernel.append(np.array([5,5,64,64]))
#kernel.append(np.array([5,5,64,64]))


input_size=np.array([None,32,32,3])#[batch, in_height, in_width, in_channels]
output_size=10
featuremap=int(32/2**num_conv)# the width/hight of the last feature map
for stride_layer in stride:
    featuremap /= stride_layer[1]
featuremap=int(featuremap)
dropout=0.5#(0,1.0]  1.0 means without dropout
iteration=50000

assert len(num_neurons) == num_fully
assert len(kernel) ==num_conv
##############################

########## Load data #########
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

image=[]
target_one=[]
for i in range(5):
    all=unpickle('cifar10/data_batch_'+str(i+1))
    data=np.float32(all[b'data']/256)#Normalize to -1~1
    RGB=np.reshape(data,[-1,3,32,32])
    RGB=np.transpose(RGB,[0,2,3,1])#[batch, in_height, in_width, in_channels]
    labels=np.int32(all[b'labels'])
    group=int(10000/num_data_batch)
    one_hot = np.zeros([10000,output_size],dtype=np.float32)
    one_hot[np.arange(10000),np.int32(labels)]=1.0

    for j in range(group):
           image.append(RGB[j*num_data_batch:(j+1)*num_data_batch,:])
           target_one.append(one_hot[j*num_data_batch:(j+1)*num_data_batch])
    RGB=np.flip(RGB,2)  #data augmentation -> flip horizontally
    for j in range(group):
           image.insert(0,RGB[j*num_data_batch:(j+1)*num_data_batch,:])
           target_one.insert(0,one_hot[j*num_data_batch:(j+1)*num_data_batch])
#testing

all=unpickle('cifar10/test_batch')
data=np.float32(all[b'data']/256)
RGB=np.reshape(data,[-1,3,32,32])
RGB=np.transpose(RGB,[0,2,3,1])#[batch, in_height, in_width, in_channels]
labels=np.int32(all[b'labels'])
one_hot = np.zeros([10000,output_size],dtype=np.float32)
one_hot[np.arange(10000),np.int32(labels)]=1.0
image_t=RGB
target_one_t=one_hot

#############################
mymodel= CNN.model(num_conv,num_fully,num_neurons,stride,kernel,input_size,output_size,featuremap)

with tf.Session() as sess:

    saver = tf.train.Saver(max_to_keep=10)
    writer = tf.summary.FileWriter(save_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        number=int(i%total_batch)
        x_in=image[number]
        ys_one=target_one[number]
        _,loss,merge,acu=sess.run([mymodel.optimize,mymodel.cross_entropy,mymodel.sum_train,mymodel.accuracy],feed_dict={mymodel.xs:x_in,mymodel.ys_one:ys_one,mymodel.drop:dropout})
        writer.add_summary(merge, i)
        if i%20==0:
           print('Iteration:{0}  Loss:{1}  Training accuracy:{2}'.format(i,loss,acu))

        if i%100==0:
            accu, merge = sess.run([mymodel.accuracy, mymodel.sum_test],
                                   feed_dict={mymodel.xs: image_t,mymodel.ys_one:target_one_t,
                                              mymodel.drop: 1.0})
            print("Iteration:{0}  Testing accuracy:{1}".format(i, accu))
            writer.add_summary(merge, i)
            print('Save model to:saver/my-model-{}'.format(iteration))
            saver.save(sess, save_dir + '/my-model', global_step=iteration)
