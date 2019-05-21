'comment: for v2, useful data start from 5th element; while for tiny-v2, useful data start from 6th.'

import numpy as np
import tensorflow as tf
import cv2

imgSrc = cv2.imread('sample_person.jpg', 1)
(H, W, _) = imgSrc.shape
img = np.float32(cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)/255)
imgSize = 416
img = cv2.resize(img, (imgSize, imgSize))[np.newaxis, :,:,:]
tiny = np.fromfile('yolov2.weights', np.float32)[4:]
anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]).reshape([5,2])

def decode(detection_feat, feat_sizes=(13, 13), num_classes=80, anchors=None):
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
    # shape(-1,169,5)+shape(1,169,1)
    bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
    bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5
    # shape(-1,169,5,4)
    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h, bbox_x + bbox_w, bbox_y + bbox_h], axis=3)
    return bboxes, obj_probs, class_probs

def conv2d(x, infilters, outfilters, name, ind, tiny, size=3, stride=1, batchnorm=True):
    if batchnorm == True:
        beta, gamma, mean, var = tiny[ind:ind+4*outfilters].reshape([4, outfilters])##
        ind = ind+4*outfilters
    else:
        b = tiny[ind:ind+outfilters]
        bias = tf.Variable(b, name = 'b'+name)
        ind = ind + outfilters
    num = size*size*infilters*outfilters
    w = np.transpose(tiny[ind:ind+num].reshape([outfilters, infilters, size, size]), (2,3,1,0))
    Weights = tf.Variable(w, name = 'W'+name)
    ind = ind + num
    if batchnorm == True:
        xx = tf.nn.conv2d(x, Weights, [1,stride,stride,1], 'SAME')
        xx = tf.contrib.layers.batch_norm(xx, scale=True, param_initializers={'beta':tf.constant_initializer(beta), 'gamma':tf.constant_initializer(gamma), 'moving_mean':tf.constant_initializer(mean), 'moving_variance':tf.constant_initializer(var)}, is_training=False)
        return tf.nn.leaky_relu(xx, 0.1), ind
    else:
        return tf.nn.conv2d(x, Weights, [1, stride,stride,1], 'SAME') + bias, ind
def max_pool(x, size=2, stride=2):
    return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'VALID')

ind = 0
x = tf.placeholder(tf.float32, [None,imgSize,imgSize,3])

net, ind = conv2d(x, 3, 32, '_conv1', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 32, 64, '_conv2', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 64, 128, '_conv3', ind, tiny)
net, ind = conv2d(net, 128, 64, '_conv4', ind, tiny, size=1)
net, ind = conv2d(net, 64, 128, '_conv5', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 128, 256, '_conv6', ind, tiny)
net, ind = conv2d(net, 256, 128, '_conv7', ind, tiny, size=1)
net, ind = conv2d(net, 128, 256, '_conv8', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 256, 512, '_conv9', ind, tiny)
net, ind = conv2d(net, 512, 256, '_conv10', ind, tiny, size=1)
net, ind = conv2d(net, 256, 512, '_conv11', ind, tiny)
net, ind = conv2d(net, 512, 256, '_conv12', ind, tiny, size=1)
net, ind = conv2d(net, 256, 512, '_conv13', ind, tiny)
shortcut = net
net = max_pool(net)
net, ind = conv2d(net, 512, 1024, '_conv14', ind, tiny)
net, ind = conv2d(net, 1024, 512, '_conv15', ind, tiny, size=1)
net, ind = conv2d(net, 512, 1024, '_conv16', ind, tiny)
net, ind = conv2d(net, 1024, 512, '_conv17', ind, tiny, size=1)
net, ind = conv2d(net, 512, 1024, '_conv18', ind, tiny)
net, ind = conv2d(net, 1024, 1024, '_conv19', ind, tiny)
net, ind = conv2d(net, 1024, 1024, '_conv20', ind, tiny)
shortcut, ind = conv2d(shortcut, 512, 64, '_shortcut', ind, tiny, size=1)
shortcut = tf.extract_image_patches(shortcut, [1,2,2,1], [1,2,2,1], [1,1,1,1], 'VALID')

net = tf.concat([shortcut, net], axis=-1)
net, ind = conv2d(net, 1280, 1024, '_conv21', ind, tiny)
out, ind = conv2d(net, 1024, 425, '_conv22', ind, tiny, size=1, batchnorm=False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

bbox, obj_probs, class_probs = sess.run(decode(out, feat_sizes=(13,13), anchors=anchors), feed_dict={x:img})

class_probs = np.max(class_probs, axis = 3).reshape([-1])
obj_probs = obj_probs.reshape([-1])
bbox = bbox.reshape([-1,4])
confidence = class_probs * obj_probs
confidence[confidence<0.5] = 0

det_indTotal = []
while (np.max(confidence)!=0):
    det_ind = np.argmax(confidence)
    bbox1_area = (bbox[det_ind,3]-bbox[det_ind,1])*(bbox[det_ind,2]-bbox[det_ind,0])
    sign = 0
    for coor in det_indTotal:
        if coor == None:
            det_indTotal.append(det_ind)
        else:
            xi1 = max(bbox[det_ind, 0], bbox[coor, 0])
            yi1 = max(bbox[det_ind, 1], bbox[coor, 1])
            xi2 = min(bbox[det_ind, 2], bbox[coor, 2])
            yi2 = min(bbox[det_ind, 3], bbox[coor, 3])
            int_area = (yi2-yi1)*(xi2-xi1)
            bbox2_area = (bbox[coor,3]-bbox[coor,1])*(bbox[coor,2]-bbox[coor,0])
            uni_area = bbox1_area + bbox2_area - int_area
            iou = int_area/uni_area
            if iou>0.5:
                sign = 1
                break
    if sign==0:
        det_indTotal.append(det_ind) 
    confidence[det_ind] = 0
    
depict = []
for det_ind in det_indTotal:
    x1,y1,x2,y2 = bbox[det_ind]
    x1 = int(x1*W); x2 = int(x2*W); y1 = int(y1*H); y2 = int(y2*H)
    depict.append([x1,y1,x2,y2])
    cv2.rectangle(imgSrc, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.imshow('res', imgSrc)
