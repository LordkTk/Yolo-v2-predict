# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:46:20 2019

@author: cfd_Liu
"""

'comment: the img format must be RGB, or the results will be imperfect; the img must be normalized to 0-1, which means img/255 is needed.'
'for the max_pool layer with size=2 and stride=1, argument should be tf.nn.max_pool(-,-,[1,1,1,1], padding = SAME) to make sure img size is 13*13'
'for the last convolutional layer with size=1, stride=1 and pad=1, argument should be tf.nn.conv2d(-,-,[1,1,1,1], padding = SAME) to keep the img size. And this means no padding is done in fact.'

import numpy as np
import tensorflow as tf
import cv2
imgSrc = cv2.imread('sample_person.jpg', 1)
(H, W, _) = imgSrc.shape
img = np.float32(cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)/255)
img = cv2.resize(img, (416, 416))[np.newaxis, :,:,:]
tiny = np.fromfile('yolov2-tiny.weights', np.float32)[5:]
anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]).reshape([5,2])

def decode(detection_feat, feat_sizes=(13, 13), num_classes=80,
           anchors=None):
    """decode from the detection feature"""
    H, W = feat_sizes
    num_anchors = len(anchors)
    # detetion_results shape(-1,169,5,85)
    detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors, num_classes + 5])

    bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, :2])
    bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])
    obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])
    class_probs = tf.nn.softmax(detetion_results[:, :, :, 5:])

    anchors = tf.constant(anchors, dtype=tf.float32)

    # bboxes->(x1,y1,x2,y2)
    height_ind = tf.range(H, dtype=tf.float32)
    width_ind = tf.range(W, dtype=tf.float32)
    # x_offset:shape(13,13)
    x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
    # shape(1,169,1)
    x_offset = tf.reshape(x_offset, [1, -1, 1])
    y_offset = tf.reshape(y_offset, [1, -1, 1])

    # decode
    # shape(-1,169,5)+shape(1,169,1)
    bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
    bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5

    # shape(-1,169,5,4)
    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h, bbox_x + bbox_w, bbox_y + bbox_h], axis=3)

    return bboxes, obj_probs, class_probs

def batch_norm(x, gamma, beta, mean, var):
    return tf.contrib.layers.batch_norm(x, scale=True, param_initializers={'beta':tf.constant_initializer(beta), 'gamma':tf.constant_initializer(gamma), 'moving_mean':tf.constant_initializer(mean), 'moving_variance':tf.constant_initializer(var)}, is_training=False)
def weight_bn(tiny, ind, infilters, outfilters, size, name):
    beta, gamma, mean, var = tiny[ind:ind+4*outfilters].reshape([4, outfilters])##
    ind = ind+4*outfilters
    num = size*size*infilters*outfilters
    w = np.transpose(tiny[ind:ind+num].reshape([outfilters, infilters, size, size]), (2,3,1,0))
    Weights = tf.Variable(w, name = name)
    return Weights, gamma, beta, mean, var, ind+num
def weight_bias(tiny, ind, infilters, outfilters, size, name):
    num = outfilters
    b = tiny[ind:ind+num]
    bias = tf.Variable(b, name = 'b'+name)
    ind = ind + num
    num = size*size*infilters*outfilters
    w = np.transpose(tiny[ind:ind+num].reshape([outfilters, infilters, size, size]), (2,3,1,0))
    Weights = tf.Variable(w, name = 'W'+name)
    return Weights, bias, ind+num
ind = 0
x = tf.placeholder(tf.float32, [None,416,416,3])
leak = 0.1
'conv1'
W_conv1, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 3, 16, 3, 'W_conv1')
bn = batch_norm(tf.nn.conv2d(x, W_conv1, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv1 = tf.nn.leaky_relu(bn, leak)
pool = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], 'VALID')
'conv2'
W_conv2, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 16, 32, 3, 'W_conv2')
bn = batch_norm(tf.nn.conv2d(pool, W_conv2, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv2 = tf.nn.leaky_relu(bn, leak)
pool = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], 'VALID')
'conv3'
W_conv3, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 32, 64, 3, 'W_conv3')
bn = batch_norm(tf.nn.conv2d(pool, W_conv3, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv3 = tf.nn.leaky_relu(bn, leak)
pool = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], 'VALID')
'conv4'
W_conv4, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 64, 128, 3, 'W_conv4')
bn = batch_norm(tf.nn.conv2d(pool, W_conv4, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv4 = tf.nn.leaky_relu(bn, leak)
pool = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], 'VALID')
'conv5'
W_conv5, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 128, 256, 3, 'W_conv5')
bn = batch_norm(tf.nn.conv2d(pool, W_conv5, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv5 = tf.nn.leaky_relu(bn, leak)
pool = tf.nn.max_pool(conv5, [1,2,2,1], [1,2,2,1], 'VALID')
'conv6'
W_conv6, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 256, 512, 3, 'W_conv6')
bn = batch_norm(tf.nn.conv2d(pool, W_conv6, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv6 = tf.nn.leaky_relu(bn, leak)
pool = tf.nn.max_pool(conv6, [1,2,2,1], [1,1,1,1], 'SAME')
#pool = conv6
'conv7'
W_conv7, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 512, 1024, 3, 'W_conv7')
bn = batch_norm(tf.nn.conv2d(pool, W_conv7, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv7 = tf.nn.leaky_relu(bn, leak)
'conv8'
W_conv8, gamma, beta, mean, var, ind = weight_bn(tiny, ind, 1024, 512, 3, 'W_conv8')
bn = batch_norm(tf.nn.conv2d(conv7, W_conv8, [1,1,1,1], 'SAME'), gamma, beta, mean, var)
conv8 = tf.nn.leaky_relu(bn, leak)
'conv9'
W_conv9, b_conv9, ind = weight_bias(tiny, ind, 512, 425, 1, '_conv9')
out = tf.nn.conv2d(conv8, W_conv9, [1,1,1,1], 'SAME') + b_conv9


sess = tf.Session()
sess.run(tf.global_variables_initializer())

bbox, obj_probs, class_probs = sess.run(decode(out, feat_sizes=(13,13), anchors=anchors), feed_dict={x:img})

class_probs = np.max(class_probs, axis = 3).reshape([-1])
obj_probs = obj_probs.reshape([-1])
bbox = bbox.reshape([-1,4])
confidence = class_probs * obj_probs
confidence[confidence<0.5] = 0
indTotal = []
while (np.max(confidence)!=0):
    ind = np.argmax(confidence)
    bbox1_area = (bbox[ind,3]-bbox[ind,1])*(bbox[ind,2]-bbox[ind,0])
    sign = 0
    for coor in indTotal:
        if coor == None:
            indTotal.append(ind)
        else:
            xi1 = max(bbox[ind, 0], bbox[coor, 0])
            yi1 = max(bbox[ind, 1], bbox[coor, 1])
            xi2 = min(bbox[ind, 2], bbox[coor, 2])
            yi2 = min(bbox[ind, 3], bbox[coor, 3])
            int_area = (yi2-yi1)*(xi2-xi1)
            bbox2_area = (bbox[coor,3]-bbox[coor,1])*(bbox[coor,2]-bbox[coor,0])
            uni_area = bbox1_area + bbox2_area - int_area
            iou = int_area/uni_area
            if iou>0.5:
                sign = 1
                break
    if sign==0:
        indTotal.append(ind) 
    confidence[ind] = 0
depict = []
for ind in indTotal:
    x1,y1,x2,y2 = bbox[ind]
    x1 = int(x1*W); x2 = int(x2*W); y1 = int(y1*H); y2 = int(y2*H)
    depict.append([x1,y1,x2,y2])
    cv2.rectangle(imgSrc, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.imshow('res', imgSrc)
