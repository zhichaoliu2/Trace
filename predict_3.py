
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import tensorflow as tf
import math

import numpy as np
import sys
import random

import matplotlib
matplotlib.use('Agg')  ## Avoid some problem when running on Windows or so..
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans

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

############ Deep Learning Training Process#########################
def predict(imgs_raw, pk_list, RESULTS_PATH, K_means = None, PLOT_IMG = None):

    images0 = np.loadtxt('Imgs_mean_std.txt')
    mean_img = images0[0]
    std_img = images0[1]

    pred_mat0 = np.copy(imgs_raw)
    for i0 in pred_mat0:
        i0 *= 255.0/i0.max()    ## Normalize the image to gray scale

    #pred_mat = np.copy(pred_mat0)
    out_array=[]
    for ii, pixel in enumerate(np.transpose(pred_mat0)):
        tmp2 = [(k-mean_img[ii])/(std_img[ii]+epsilon) for k in pixel]
        out_array.append(tmp2)
    pred_mat = np.transpose( out_array )
    ## Now the prediction matrix created!  Predict!

    learning_rate = 0.0001

    x = tf.placeholder(tf.float32, [None, 60 * 12])

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
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    score_save = []
    num_models = 10
    for jj in range(num_models):
        
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint('./models/'))   
        saver.restore(sess, './pre-trained_models/model' +str(jj) )   
        
        print ('TF model ' + str(jj) + ' loaded done!! ')

        cc = 0.0
        sss = []
        resultss = []
        resultss = y_conv.eval(feed_dict={x: pred_mat, keep_prob: 1.0})
        for kk in range(len(resultss)):
            diff = round(resultss[kk][0] - resultss[kk][1], 1)
            #if diff*(labels[kk][0]-0.5) > 0:
            # sss.append([int(diff>0)])
            if diff > 0.0:
                sss.extend([1])
                cc+=1.0
            else:
                sss.extend([0])

        score_save.append(sss)
        print ('Model ' + str(jj) + ' Predicted peaks: ', cc, ' from ', len(pk_list), 'target images' )
        sess.close()

    score_vote = np.mean(np.transpose(score_save), axis = 1)
    target_pks = []
    target_imgs = []
    for kk, skk in enumerate(score_vote):
        if skk > 0.5:
            tmp2 = []
            tmp2.extend(pk_list[kk])
            tmp2.append(skk)
            target_pks.append(tmp2)
            target_imgs.append(imgs_raw[kk])

    print ('Final peaks predicted: ', len(target_pks))
    f2 = (RESULTS_PATH + "/ImageData_Final-pks.txt")
    np.savetxt(f2, target_imgs, fmt='%.2f',delimiter=' ')

    if PLOT_IMG :
        os.system('mkdir ./Results/Signal_Images')
        print ('Now Ploting...') 
        for kk in range(np.shape(target_pks)[0]):
            mz0 = round(tmp2[0], 3); rt0 = round(tmp2[1], 3)
            plt.imshow(np.reshape(imgs_raw[kk], (60, 12)), interpolation='bilinear', cmap='jet', aspect='auto')

            plt.title("M/Z: " + str(mz0)+"  RT: " +str(rt0) )
            plt.xlabel('M/Z')
            plt.ylabel('Time')
            plt.colorbar()
            plt.savefig(RESULTS_PATH+ "/Signal_Images/Signal_" + str(kk+1) + '_' + str(mz0) + '_'+ str(rt0) + '.png')
            plt.clf()

    if K_means :

        for i0 in target_imgs:
            i0 *= 255.0/i0.max()    ## Normalize the image to gray scale

        kmeans = KMeans(n_clusters= int(K_means), random_state=0).fit(target_imgs)
        k_label = np.array(kmeans.labels_)
        k_center = np.array(kmeans.cluster_centers_)

        rows = math.ceil(K_means/3.0)
        plt.figure()
        for j in range(K_means):
            plt.subplot(rows, 3, j+1)
            plt.imshow(np.reshape(k_center[j], (60, 12)), interpolation='bilinear', cmap='jet', aspect='auto')

            plt.title('Cluster ' + str(j+1))
        plt.savefig( RESULTS_PATH+ "/Clusters.png")
        plt.clf()

        for j in range(np.shape(target_pks)[0]):
            target_pks[j].extend([k_label[j]])

    f1 = RESULTS_PATH+ "/Final_pks.txt"
    np.savetxt(f1, target_pks, fmt='%.3f',delimiter=' ')

    return target_pks

