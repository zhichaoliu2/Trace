
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import random
import os


MODEL_PATH = "./New-trained_models"
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

train_ratio = 0.8

############ Deep Learning Training Process#########################
images0 = np.loadtxt('imgs-train.txt')   ## N*m matrix, m = 60*12, stands for a flatted image
labels = np.loadtxt('label-train.txt')

if np.shape(labels)[1]==1 :
    labelt = np.transpose(labels)
    labels = np.transpose(np.concatenate((at, [1-at0 for at0 in labelt])))

for bb0 in images0:    ### Normalize to be 0~255.
    bb0 *= (255.0/bb0.max() )

mean_img = []
std_img = []
## batch normalize the images according to pixel and save the mean and std of each pixel ##
epsilon = 0.001
out_array=[]
for pixel in np.transpose(images0):
    mean_v = np.mean(pixel)
    std_v  = np.std(pixel)
    tmp2 = [(k-mean_v)/(std_v+epsilon) for k in pixel]
    mean_img.append(mean_v)
    std_img.append(std_v)
    out_array.append(tmp2)

images = np.transpose(out_array)
#######################################


# Weight Initialization
def weight_variable(shape, var_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = var_name )

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 60 * 12])
################## LeNet5 #################
# First Convolutional Layer
W_conv1 = weight_variable([4, 4, 1, 32],"W_conv1")    ## 4*4*1*32
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,60,12,1])  # 60, 12

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([4, 4, 32, 64] ,"W_conv2")   ## 4*4*32*64
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([15 * 3 * 64, 256],"W_fc1")   #  ,1024
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 15*3*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([256, 2],"W_fc2")
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
################### LeNet5 ################

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_models = 5
steps = 5000
batch_size = 64

saver = tf.train.Saver(max_to_keep = num_models)   ##@@ Save the trained model @@##

for jj in range(num_models):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, train_size= train_ratio )
    
    for i in range(steps):
        random_select =  np.random.randint(0, len(labels_train), batch_size)
        batch_x = [ images_train[j1] for j1 in random_select ]
        batch_y = [ labels_train[j2] for j2 in random_select ]
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        if i==0 or (i+1)%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={ x:images_train, y_: labels_train, keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={ x: images_test, y_: labels_test, keep_prob: 1.0})
            print(i+1, train_accuracy, test_accuracy )

    saver.save(sess, 'models-lenet5/model' + str(jj))
    
    print (" Saved Machine :", jj, )
    print (" Train_accuracy: ", train_accuracy, "  Test_accuracy: ",  test_accuracy )


