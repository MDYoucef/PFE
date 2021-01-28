from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import cPickle
import os
import glob
import joblib as jl
import random
import copy
#%matplotlib inline

# In[2]:

def cifar10_concatenate():
    
    img = np.zeros([50000,3072])
    lbl = np.zeros([50000])
    for i in range(5):
        with open('/home/skyolia/TF_Project/Cifar10/cifar-10-batches-py/data_batch_'+str(i+1),'rb') as f:
            data = cPickle.load(f)
        for j in range(10000):
            img[j+10000*i] = data['data'][j]
            lbl[j+10000*i] = data['labels'][j]
            
    with open('/home/skyolia/TF_Project/Cifar10/cifar-10-batches-py/test_batch','rb') as f:
        data2 = cPickle.load(f)
        test_labels = np.asarray(data2['labels'])
        test_data = np.asarray(data2['data'])
        
    return img, lbl, test_data, test_labels

#train_x, train_y, test_x, test_y = cifar10_concatenate()


# In[3]:
'''
def rand_cropping(data):
    
    cropp = list()
    shape = data.shape[0]
    
    for i in range(shape):
        
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        cropped = data[i][x: x + 32, y: y + 32, :]
        cropp.append(cropped.flatten()) 
    return np.asarray(cropp)
'''

def rand_flip(data): #rgb input
    
    x = np.random.uniform(0,1,1)
    if x <= 0.5:
        data = np.flip(data,1)
    return data
    
def rand_cropping_flat(data):
    
    x = np.random.randint(0, 8)
    y = np.random.randint(0, 8)
    cropped = data[x: x + 32, y: y + 32, :]
    return cropped.flatten()

def rand_cropping_flipping(data):
    
    augmented = list()
    shape = data.shape[0]
    for i in range(shape):
        
        img = data[i]
        flip = rand_flip(img)
        crop = rand_cropping_flat(flip)
        augmented.append(crop)
        
    return np.asarray(augmented)

def pad(data, p):
    
    #x = np.random.randint(0, 8)
    #y = np.random.randint(0, 8)
    
    padding = np.pad(data, ((p, p),(p, p),(0,0)), 'constant', constant_values=(0,0))
    data = padding
    #cropping = padding[x: x + data_shape[0], y: y + data_shape[1], :]
        
    return data


# In[5]:

def cifar10_reshape_augment(data, p, augment):
    
    size = data.shape[0]
    raw = list()
    new = list()
    
    for i in range(size):
        
        imageToUse = data[i] / 255.0
        image = imageToUse.reshape(3,32,32).transpose(1,2,0)
        raw.append(image.flatten())
        if augment:
            
            augmented = pad(image, p)
            new.append(augmented)       
      
    return np.asarray(raw), np.asarray(new)


# In[ ]:

def cifar10_preparation(xtrain, xtest, train_y, test_y, augment):
    
    train_x, rgb = cifar10_reshape_augment(xtrain, 4, augment)
    print("train done")
    test_x, _ = cifar10_reshape_augment(xtest, 0, False)
    print("test done")
    
    return train_x, train_y, test_x, test_y, rgb

#train_x, train_y, test_x, test_y, _ = cifar10_preparation(train_x, test_x, train_y, test_y, False)

def re(data):
    
    size = data.shape[0]
    raw = list()
    for i in range(size):
        
        image = data[i].reshape(32,32,3)
        aug = pad(image, 4)
        raw.append(aug)
    
    return np.asarray(raw)

#train_x =re(train_x)




# In[ ]:
'''
jl.dump(train_x, 'train_x.pkl')
jl.dump(train_y, 'train_y.pkl')
jl.dump(test_x, 'test_x.pkl')
jl.dump(test_y, 'test_y.pkl')
jl.dump(rgb, 'rgb.pkl')
print('done')


# In[2]:

train_x = jl.load('train_x.pkl')
print(train_x.shape)
train_y = jl.load('train_y.pkl')
print(train_y.shape)
test_x = jl.load('test_x.pkl')
print(test_x.shape)
test_y = jl.load('test_y.pkl')
print(test_y.shape)
rgb = jl.load('rgb.pkl')
print(rgb.shape)
'''
# In[3]:

def shuffle(train_x, train_y):
    
    #ran = np.random.randint(0, size, size=size)
    size = train_y.shape[0]
    ran = random.sample(range(size), size)
    data = list()
    label = list()
    j=0
    for i in ran:
        data.insert(i,train_x[j])
        label.insert(i,train_y[j])
        j+=1
    
    return np.asarray(data), np.asarray(label)


# In[21]:

def plotCifar(data, label, size, r, c, reshape):
    
    fig = plt.figure(figsize=(30, 15))
    ex = random.sample(range(size), r*c)
    for i in range(r*c):
        
        ax = fig.add_subplot(r, c, i+1, xticks=[], yticks=[])
        ax.set_title(str(label[ex[i]]))
        img = data[ex[i]]
        if reshape:
            img = img.reshape(32,32,3)
        plt.imshow(img)
    
    plt.show()

#plotCifar(test_x, test_y, 10000, 5, 3)
   
'''
aug_x, aug_y = shuffle(rand_cropping(rgb), train_y, size = 50000)
#aug_batch = create_batches(128,aug_x, aug_y)
ran = plotNNFilter(aug_x, aug_y, 50000, 30)
print(rand_cropping(rgb).shape)
'''

# In[4]:

def batch_norm(x, n_out, phase_train, convolutional = False, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        if convolutional:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def weight(shape, name):
    
    w_name = 'w_'+name
    print("shape", shape)
    return tf.get_variable(name=w_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def conv_bn_elu(x, i, name):
        
    in_d = x.get_shape().as_list()[3]
    out_d = depth[i]
    shape = [3, 3, in_d, out_d]
    weights = weight(shape, name)
    
    conv_name = 'conv_'+name
    conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME', name=conv_name)
    bn = batch_norm(conv, out_d, phase_train, convolutional = True)
    elu = tf.nn.elu(bn)
    print("elu", elu.get_shape())
    print("_"*50)
    
    return elu
    
def pool(x):
    
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    return pool

def dropout(x, prob):
    
    drop = tf.nn.dropout(x, keep_prob=prob)
    
    return drop

def gap(x):
    
    
    global_pool = tf.reduce_mean(x, [1, 2])
    in_d = x.get_shape().as_list()[3]
    shape = [in_d, 10]
    name = 'gap'
    weights = weight(shape, name)
    
    biais = tf.Variable(tf.constant(0.1, shape=[10]))
    out = tf.matmul(global_pool, weights) + biais
    print("gap", global_pool.get_shape())
    print('_'*50)
    print("out", out.get_shape())
    
    return out
    
def create_batches(batch_size, data, label):
    
    while (True):
        for i in xrange(0, len(label), batch_size):
            yield(data[i:i+batch_size],label[i:i+batch_size])
       
        


# In[5]:
'''
tf.reset_default_graph()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 3072], name="x_input")
    y = tf.placeholder(tf.int64, shape=[None], name="y_input")
    prob_1=tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)
    
depth = [48,96,192]
inp = tf.reshape(x,[-1,32,32,3])
inp = batch_norm(inp, 3, phase_train, convolutional = True)
pooling = False
for i in range(3):
    for j in range(2):
        name = str(i)+'_'+str(j)
        hidden=conv_bn_elu(inp, i, name)
        inp = hidden
    hidden = pool(inp)
    inp = dropout(hidden, prob_1)

hidden = conv_bn_elu(inp, i, '3_0')
inp = dropout(hidden, prob_1)
out_y = gap(inp)


# In[7]:

logs_path = "/home/skyolia/tensorflow_project/cifar-10/CNN/APRES/Data augmentation/model_2/"
learning_rate = 1e-3

with tf.name_scope('cross_entropy'):
    
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out_y, y))
        
with tf.name_scope('learning_rate'):
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
with tf.name_scope('Accuracy'):
    
    correct_pred = tf.equal(tf.argmax(out_y, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

acc_test_summary = tf.summary.scalar("test_accuracy", accuracy)

lost_training_summary = tf.scalar_summary("training_lost", cost)
lost_test_summary = tf.scalar_summary("test_lost", cost)

writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
saver = tf.train.Saver(max_to_keep=300)

epoch = 0
raw_batch = create_batches(128,train_x, train_y)
raw_batch2 = create_batches(100,test_x, test_y)
start = time.time()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init, {phase_train: True})
    while(True):
        
        acc, acc2, c, c2 = 0.0, 0.0, 0.0, 0.0
        print("epoch : ", epoch)
        #shuf_x, shuf_y = shuffle(rgb, train_y, 50000)
        aug_batch = create_batches(128,rgb, train_y)
        
        for j in range(391):
            
            img_aug, lbl_aug = aug_batch.next()
            aug_x = rand_cropping(img_aug)
            
            img, lbl = raw_batch.next()
            print('aug_x = ',aug_x.shape, '\nimg = ',img.shape)
            
            concat_x = np.concatenate((aug_x, img), axis = 0) #train_x
            concat_y = np.concatenate((lbl_aug, lbl), axis = 0) #train_y
            batch_x, batch_y = shuffle(concat_x, concat_y, 256)
            print('batch_x = ',batch_x.shape, '\nbatch_y = ',batch_y.shape)
            
            feed_dict={x: batch_x, y: batch_y, prob_1: 1., phase_train: True}
            optimizer.run(feed_dict = feed_dict)
            feed_dict={x: batch_x, y: batch_y, prob_1: 1., phase_train: False}
            batch_c, batch_acc = sess.run([cost, accuracy], feed_dict = feed_dict)
    
            c += batch_c
            acc += batch_acc
                    
            if (j%80 == 0):
                print("j = ",j)
                print("batch train accuracy = ", batch_acc)
                        
            train_lost_summ = sess.run(lost_training_summary, feed_dict = feed_dict)
            writer.add_summary(train_lost_summ,epoch * 391 + j)
                    
                    
        train_cost = c/391
        train_accuracy = acc/391
        end = time.time()
        duree = end-start
        print("train cost : ", train_cost, "\ntrain accuracy : ", train_accuracy, "\nduree : ", duree)
            
        
        for j in range(100):
            img2, lbl2 = raw_batch2.next()
            feed_dict={x: img2, y: lbl2, prob_1: 1., phase_train: False}
                
            acc2 += sess.run(accuracy, feed_dict = feed_dict)
            c2 += sess.run(cost, feed_dict = feed_dict)
                    
            test_acc_summ, test_lost_summ = sess.run([acc_test_summary, lost_test_summary], feed_dict = feed_dict)
            writer.add_summary(test_acc_summ,epoch * 100 + j)
            writer.add_summary(test_lost_summ,epoch * 100 + j)
            
        test_cost = c2/100
        test_accuracy = acc2/100
        print("test cost = ", test_cost, "\ntest accuracy : ", test_accuracy)
            
        file_name = "./"+str(epoch)+"_model.ckpt"
        saver.save(sess, file_name)
            
        epoch += 1 
    
print("model saved")
'''

# In[ ]:



