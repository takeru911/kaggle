import tensorflow as tf
import pandas as pd
import numpy as np
from logging import getLogger, StreamHandler, DEBUG
import cv2
import random

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

labels = pd.read_csv("input/train_master.tsv", sep = "\t")
category_master = pd.read_csv("input/master.tsv", sep = "\t")

train_master = pd.merge(labels, category_master)
print(train_master.head())

DATA_SET = []
for index, row in train_master.iterrows():
    tmp_list = []
    # for local pc
    if(index < 3000):
        logger.debug(str(index) + ": input/images/" + row.file_name)

        img = cv2.imread("input/images/" + row.file_name, cv2.IMREAD_COLOR)
        orgHeight, orgWidth = img.shape[:2]
        size = (100, 100)
        img = cv2.resize(img, size)
        img = img.flatten().astype(np.float32)/255.0
        tmp_list.append(img)
        classes_array = np.zeros(55, dtype = 'float64')
        classes_array[int(row.category_id)] = 1
        tmp_list.append(classes_array)
        DATA_SET.append(tmp_list)
    TRAIN_DATA_SIZE = int(len(DATA_SET) * 0.8)
    TRAIN_DATA_SET = DATA_SET[:TRAIN_DATA_SIZE]
    TEST_DATA_SET = DATA_SET[TRAIN_DATA_SIZE:]

MAX_EPOCH = 400
BATCH_SIZE = 50
CHANNELS = 3
NUM_CLASSES = 3
IMAGE_SIZE = 100
IMAGE_MATRIX_SIZE = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUM_CLASSES = 55

def batch_data(data_set, batch_size):
    data_set = random.sample(data_set, batch_size)
    return data_set

def devide_data_set(data_set):
    data_set = np.array(data_set)
    image_data_set = data_set[:int(len(data_set)), :1].flatten()
    label_data_set = data_set[:int(len(data_set)), 1:].flatten()

    image_ndarray = np.empty((0, 30000))
    label_ndarray = np.empty((0, 55))

    for (img, label) in zip (image_data_set, label_data_set):
        image_ndarray = np.append(image_ndarray, np.reshape(img, (1, 30000)), axis = 0)
        label_ndarray = np.append(label_ndarray, np.reshape(label, (1, 55)), axis = 0)
    return image_ndarray, label_ndarray

def conv2d(x, W) :
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x) :
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape) :
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape) :
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
        

def deepnn(x) :

    with tf.name_scope('reshape') :
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.name_scope('conv1') :
        W_conv1 = weight_variable([5, 5, CHANNELS, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') :
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') :
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') :
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') :
        W_fc1 = weight_variable([25 * 25 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout') :
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') :
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob



x = tf.placeholder(tf.float32, [None, IMAGE_MATRIX_SIZE])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
y_conv, keep_prob = deepnn(x)
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_step in range(MAX_EPOCH):
        train_data_set = batch_data(TRAIN_DATA_SET, BATCH_SIZE)
        train_image, train_label = devide_data_set(train_data_set)

        if epoch_step % BATCH_SIZE == 0:
            train_accuracy = accuracy.eval(feed_dict={x: train_image, y_: train_label, keep_prob: 1.0})
            logger.info("epoch_step %d, training accuracy %g" % (epoch_step, train_accuracy))
        logger.debug("epoch_step %d" % (epoch_step))

        train_step.run(feed_dict={x: train_image, y_: train_label, keep_prob: 0.5})
    test_image, test_label = devide_data_set(TEST_DATA_SET)
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test_image, y_: test_label, keep_prob: 1.0
    }))

