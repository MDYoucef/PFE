from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf
import time
import scipy.misc
import matplotlib as mp
import matplotlib.pyplot as plt
import itertools
import cPickle
import collections
import Image, ImageDraw
import PIL
from PIL import ImageFont
import os
import glob
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from sklearn.metrics.classification import confusion_matrix
from helper import sem_labels, diff, wrong_mnist, touch, write, combine, own_data, dataset_creation, deparserx
def batch_norm(x, n_out, phase_train, convolutional=False, scope='bn'):
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
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        
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

def cifar10_concatenate():
    
    img = np.zeros([50000, 3072])
    lbl = np.zeros([50000])
    for i in range(5):
        with open('/home/skyolia/tensorflow_project/cifar-10/cifar-10-batches-py/data_batch_' + str(i + 1), 'rb') as f:
            data = cPickle.load(f)
        for j in range(10000):
            img[j + 10000 * i] = data['data'][j]
            lbl[j + 10000 * i] = data['labels'][j]
        
    return img, lbl

with open('/home/skyolia/tensorflow_project/cifar-10/cifar-10-batches-py/test_batch', 'rb') as f:
    data2 = cPickle.load(f)
    test_labels = np.asarray(data2['labels'])
    test_data = np.asarray(data2['data'])
    
train_data, train_labels = cifar10_concatenate()
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

def cifar10_reshape(data):
    
    size = data.shape[0]
    img = np.zeros([size, 3072])
    
    for i in range(size):
        imageToUse = data[i]
        
        image = imageToUse.reshape(3, 32, 32).transpose(1, 2, 0)
        elmn = image.flatten()
        
        img[i] = elmn
        
    return img
    
def normalisation(array):
    
    array = array.astype('float32')
    array_nomalized = array / 255.0       
    return array_nomalized

def cifar10_preparation():
    
    train_reshape = cifar10_reshape(train_data)
    test_reshape = cifar10_reshape(test_data)
    print("reshape done")
    
    norm_train_data = normalisation(train_reshape)
    norm_test_data = normalisation(test_reshape)
    print("normalisation done")
    
    return norm_train_data, norm_test_data

a, b = cifar10_preparation()

def create_batches(batch_size, isTrain):
    
    while (True):
        if isTrain:
            for i in xrange(0, len(train_labels), batch_size):
                yield(a[i:i + batch_size], train_labels[i:i + batch_size])
        else:
            for i in xrange(0, len(test_labels), batch_size):
                yield(b[i:i + batch_size], test_labels[i:i + batch_size]) 
                    

tf.reset_default_graph()
embedding_size = 1024
learning_rate = 1e-3
batch_size = 100
display_step = 1
    
    #mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=logs_path + 'data', one_hot=True)
    
    # Network Parameters
n_input = 3072  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
strides=1
k=2    
sess = tf.InteractiveSession()

    # tf Graph input
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_input], name="x_input")
    y = tf.placeholder(tf.int64, shape=[None], name="y_input")
    prob_1=tf.placeholder(tf.float32)
    prob_2=tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)

    
    # Store layers weight & bias
with tf.name_scope("weights"):
        
    weights = {

    'wc1': tf.get_variable(name = "w1",shape = [3, 3, 3, 48], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wc2': tf.get_variable(name = "w2",shape = [3, 3, 48, 48], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    #'wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name = "w3"),
    'wc3': tf.get_variable(name = "w3",shape = [3, 3, 48, 96], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wc4': tf.get_variable(name = "w4",shape = [3, 3, 96, 96], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    #'wc6': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name = "w6"),
    'wc5': tf.get_variable(name = "w5",shape = [3, 3, 96, 192], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wc6': tf.get_variable(name = "w6",shape = [3, 3, 192, 192], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    #'wc8': tf.Variable(tf.truncated_normal([1, 1, 128, 128], stddev=0.1), name = "w8"),
    'wc7': tf.get_variable(name = "w7",shape = [1, 1, 192, 192], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wc8': tf.get_variable(name = "w8",shape = [1, 1, 192, 10], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
}
    
    
with tf.name_scope("biases"):
    
    biases = {
    
    'bc7': tf.Variable(tf.constant(0.1, shape=[192]), name = "b7"),   
    'bc8': tf.Variable(tf.constant(0.1, shape=[10]), name = "b8"),
}

'''
'bc1': tf.Variable(tf.constant(0.1, shape=[48]), name='b1'),
    'bc2': tf.Variable(tf.constant(0.1, shape=[48]), name = "b2"),
    #'wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name = "w3"),
    'bc3': tf.Variable(tf.constant(0.1, shape=[96]), name = "b3"),
    'bc4': tf.Variable(tf.constant(0.1, shape=[96]), name = "b4"),
    #'wc6': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name = "w6"),
    #'wc7': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name = "w7"),
    'bc5': tf.Variable(tf.constant(0.1, shape=[192]), name = "b5"),
    'bc6': tf.Variable(tf.constant(0.1, shape=[192]), name = "b6"),
    'bc7': tf.Variable(tf.constant(0.1, shape=[192]), name = "b7"),
    'bc8': tf.Variable(tf.constant(0.1, shape=[512]), name = "b8"),
    'bc9': tf.Variable(tf.constant(0.1, shape=[256]), name = "b9"),
'''

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parametes = 1
    for dim in shape:
        print(dim)
        variable_parametes *= dim.value
    print(variable_parametes)
    total_parameters += variable_parametes
print("total_parameters : ",total_parameters)
    
x_image = tf.reshape(x,[-1,32,32,3])
x_bn = batch_norm(x_image, 3, phase_train, convolutional = True)

hidden_1 = tf.nn.conv2d(x_bn, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
hidden_1_bn = batch_norm(hidden_1, 48, phase_train, convolutional = True)
hidden_1_relu = tf.nn.elu(hidden_1_bn)
print(hidden_1_relu.get_shape())

hidden_2 = tf.nn.conv2d(hidden_1_relu, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
hidden_2_bn = batch_norm(hidden_2, 48, phase_train, convolutional = True)
hidden_2_relu = tf.nn.elu(hidden_2_bn)
print(hidden_2_relu.get_shape())

pool_1 = tf.nn.max_pool(hidden_2_relu, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')
pool_1_do=tf.nn.dropout(pool_1, keep_prob=prob_2)
print(pool_1.get_shape())

hidden_3 = tf.nn.conv2d(pool_1_do, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
hidden_3_bn = batch_norm(hidden_3, 96, phase_train, convolutional = True)
hidden_3_relu = tf.nn.elu(hidden_3_bn)
print(hidden_3_relu.get_shape())

hidden_4 = tf.nn.conv2d(hidden_3_relu, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
hidden_4_bn = batch_norm(hidden_4, 96, phase_train, convolutional = True)
hidden_4_relu = tf.nn.elu(hidden_4_bn)
print(hidden_4_relu.get_shape())

pool_2 = tf.nn.max_pool(hidden_4_relu, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')
pool_2_do=tf.nn.dropout(pool_2, keep_prob=prob_2)
print(pool_2.get_shape())

hidden_5 = tf.nn.conv2d(pool_2_do, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
hidden_5_bn = batch_norm(hidden_5, 192, phase_train, convolutional = True)
hidden_5_relu = tf.nn.elu(hidden_5_bn)
print(hidden_5_relu.get_shape())

hidden_6 = tf.nn.conv2d(hidden_5_relu, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
hidden_6_bn = batch_norm(hidden_6, 192, phase_train, convolutional = True)
hidden_6_relu = tf.nn.elu(hidden_6_bn)
print(hidden_6_relu.get_shape())

pool_3 = tf.nn.max_pool(hidden_6_relu, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')
pool_3_do=tf.nn.dropout(pool_3, keep_prob=prob_1)
print(pool_3.get_shape())

hidden_7 = tf.nn.conv2d(pool_3_do, weights['wc7'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc7']
#hidden_7_bn = batch_norm(hidden_7, 192, phase_train, convolutional = True)
hidden_7_relu = tf.nn.elu(hidden_7)
hidden_7_do=tf.nn.dropout(hidden_7_relu, keep_prob=prob_1)
print(hidden_7_relu.get_shape())

hidden_8 = tf.nn.conv2d(hidden_7_do, weights['wc8'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc8']
hidden_8_relu = tf.nn.elu(hidden_8)
print(hidden_8_relu.get_shape())

gap = tf.nn.avg_pool(hidden_8_relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")
print(gap.get_shape())

out_y = tf.reshape(gap, (-1,10))
print(out_y.get_shape())

    
    # Define loss and optimizer
with tf.name_scope('cross_entropy'):
    
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out_y, y))
        
with tf.name_scope('learning_rate'):
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
with tf.name_scope('Accuracy'):
    
    correct_pred = tf.equal(tf.argmax(out_y, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

start = time.time()

test_cumulative_accuracy = 0.0
train_cumulative_accuracy = 0.0

init = tf.global_variables_initializer()

gen_batch = create_batches(125, isTrain=True)
gen_batch2 = create_batches(100, isTrain =False)

saver = tf.train.Saver()
sess.run(init, {phase_train: False})
saver.restore(sess, "./488_model.ckpt")
test_accuracy = 0.0   

'''
for j in range(100):
    img, lbl = gen_batch2.next()
    test_accuracy += sess.run(accuracy, feed_dict={x:img, y: lbl, prob_1: 1.0, prob_2: 1.0, phase_train:False})
test_cumulative_accuracy = test_accuracy/100
print("test_cumulative_accuracy : ", test_cumulative_accuracy)
'''

pred = tf.argmax(out_y, 1)

layers = [hidden_1, hidden_1_relu, hidden_2, hidden_2_relu, pool_1,
          hidden_3, hidden_3_relu, hidden_4, hidden_4_relu, pool_2,
          hidden_5, hidden_5_relu,  hidden_6, hidden_6_relu, pool_3, 
          hidden_7, hidden_7_relu, hidden_8, hidden_8_relu, gap]
    

def tensor_to_array(layer, img):
    
    array = sess.run(layer, feed_dict={x:np.reshape(img, [1, 3072], order='F'), prob_1:1.0, prob_2:1.0, phase_train:False})
    return array

def num_filter(array):
    
    num = array.shape[3]
    return num

def filter_viewer(arr, k):
    w, x, y, z = arr.shape
    f = np.empty(shape=(x, y))
    for i in range(x):
        for j in range(y):
            f[i, j] = arr[0, i, j, k]
            
    return f

def filters_img(x):
    
    for i in range(layers.__len__()):
        
        arr = x.ravel()
        temp_arr = tensor_to_array(layers[i], arr)
        
        if(i == layers.__len__() - 1):
            
            gap = temp_arr
        
        num_f = num_filter(temp_arr)
        
        for j in range(num_f):
            
            f = filter_viewer(temp_arr, j)
            ff = "layer" + str(i)
            newpath = r'/home/skyolia/tensorflow_project/cifar-10/CNN/cnn_17/filters/' + ff + '/'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            file_name = newpath + str(j) + ".png"
            scipy.misc.imsave(file_name, f)
            
    return gap

def resized_filters():
    
    l = list()
    new_size = (150, 150)
    descente = -1
    courant = 0
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 16)
    for file in sorted(os.listdir('/home/skyolia/tensorflow_project/cifar-10/CNN/cnn_17/filters')):
        path = glob.glob('/home/skyolia/tensorflow_project/cifar-10/CNN/cnn_17/filters/' + file + '')
        for i in path:
            for j in sorted(os.listdir(i)):
                l.append(j)
            lsorted = sorted(l, key=lambda x: int(os.path.splitext(x)[0]))
            l = []
            lenth = int(np.sqrt(len(lsorted))) + 1
            ult_size = lenth * 150
            
            big_im = Image.new("RGB", (ult_size, ult_size))
            
            for k in range(len(lsorted)):
                fp = i + '/' + lsorted[k]
                newpath = r'/home/skyolia/tensorflow_project/cifar-10/CNN/cnn_17/filters_resized/' + file
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                old_im = Image.open(fp)
                image = old_im.resize((100, 100), PIL.Image.ANTIALIAS)
                old_size = image.size
                new_im = Image.new("RGB", new_size, (255, 255, 255))
                new_im.paste(image, (0, 0))
                draw = ImageDraw.Draw(new_im)
                draw.text((20, 120), "filter :" + str(k), (0, 0, 0), font=font)
                
                if k % (ult_size / 150) == 0:
                    descente += 1
                    courant = 0
                    big_im.paste(new_im, (courant, descente * 150))
                    
                else:
                    # print(descente)
                    courant += 150
                    # print(courant)
                    big_im.paste(new_im, (courant, descente * 150))
            
            descente = -1
            courant = 1   
            
                
            file_name = newpath + ".png"
            big_im.save(file_name)               
                
                
            del draw
            
def test(data):
    
    classification = sess.run(pred, feed_dict={x: [data], prob_1: 1.0, prob_2:1.0, phase_train:False})
    print ('NN predicted', classification[0])
    return sem_labels(classification[0])

def my_data(img_file, size):
    
    img = Image.open(img_file)
    plt.show()
    w, h = img.size
    r = w / h
    print(w)
    print(h)
    print(r)
    prediction_array = []
    image = img.resize((32, 32), PIL.Image.ANTIALIAS)
    
    im_rgb = image.convert('RGB')
            
    im_rgb = (np.array(im_rgb))
    
    r = im_rgb[:, :, 0].flatten()
    g = im_rgb[:, :, 1].flatten()
    b = im_rgb[:, :, 2].flatten()
            
    fdata = np.array(list(r) + list(g) + list(b), np.uint8)
    
    norm_fdata = normalisation(fdata)
            
    image = norm_fdata.reshape(3, 32, 32).transpose(1, 2, 0)
    print(image.shape)
    norm_fdata = image.flatten()
            
    prediction_array.append(sess.run(tf.argmax(out_y, 1), feed_dict={ x: [norm_fdata], prob_1: 1.0, prob_2:1.0, phase_train:False}))
    classification = np.array(prediction_array).ravel()
    return sem_labels(classification[0]), im_rgb, norm_fdata

def true_label(x, set):
    
    if set == "train":
        y = sem_labels(train_labels[x])
    elif set == "test":
        y = sem_labels(test_labels[x])
    return y

def curr_img(x, set):
    
    if set == "train":
        y = a[x]
    elif set == "test":
        y = b[x]
    return y

def test_classification(set): 
    
    if set == "train":
        
        predicted_array=[]
        print('ici ',set)
        for i in xrange(400):
            img, lbl = gen_batch.next()
            predicted_array.append(sess.run(pred, feed_dict={ x: img, prob_1: 1.0, prob_2:1.0, phase_train:False}))
            desired_labels = train_labels
            data = a 
            
    
    elif set == "test":
        predicted_array=[]
        for i in xrange(100):
            img, lbl = gen_batch2.next()
            predicted_array.append(sess.run(pred, feed_dict={ x: img, prob_1: 1.0, prob_2:1.0, phase_train:False}))
            desired_labels = test_labels
            data = b
        
    print(set)
    predicted_labels = np.array(predicted_array).ravel()
    size = desired_labels.shape[0]
    print(len(predicted_labels))
    
    print(len(desired_labels))
    
    wrong = diff(desired_labels, predicted_labels)
    
    size = size + 0.00
    acc = ((size - len(wrong))/size)*100
    print("accuracy : " , acc)
    print(wrong)
    print("size :")
    print(len(wrong))
    
    sav = wrong_mnist(wrong, data, set)
    t, path=touch(set)
    tester=write(set, path, t, desired_labels, predicted_labels, wrong)
    ultim=combine(tester, set)
    
    
    
    cm = confusion_matrix(desired_labels,predicted_labels)
    
    return acc,cm

                
            

        
