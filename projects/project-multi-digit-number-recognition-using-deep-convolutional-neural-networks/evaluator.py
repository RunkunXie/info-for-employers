# -*- coding: utf-8 -*-

"""
Created on Tue Dec 10 12:03:31 2019
@author: User
"""

import tensorflow as tf
from model import Model
from preproc import DataManager

def evaluate(main_path, path_to_latest_checkpoint_file, path_to_validation_tfrecord, dataset_type, batch_size=128, drop_rate=0.0):
    
    # get validation data
    dm = DataManager(main_path, dataset_type=dataset_type)
    num_val = dm.n_samps
    image_batch, length_batch, digits_batch = dm.data_generator(path_to_validation_tfrecord, batch_size, is_shuffle=False)  
    
    # feed image data into model
    length_prob, digits_prob = Model.inference(image_batch, drop_rate=drop_rate)

    length_pred = tf.argmax(length_prob, axis=1)
    digits_pred = tf.argmax(digits_prob, axis=2)
    
    # the following four lines of codes are referenced from online resources
    labels = tf.concat([tf.reshape(length_batch, [-1,1]), digits_batch], axis=1)
    labels = tf.reduce_join(tf.as_string(labels), axis=1)
    
    # get prediction
    predictions = tf.concat([tf.reshape(length_pred, [-1,1]), digits_pred], axis=1)
    predictions = tf.reduce_join(tf.as_string(predictions), axis=1)
    
    # get validation accuracy
    acc, update_op = tf.metrics.accuracy(labels=labels,
                                         predictions=predictions)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # restore the pre-trained model

        saver = tf.train.Saver()
        saver.restore(sess, path_to_latest_checkpoint_file)
        
        # update the local variables by evaluating each batch
        for i in range(num_val // batch_size):
            sess.run(update_op)

        accuracy = sess.run(acc)
        coord.request_stop()
        coord.join(threads)

    return accuracy

    



 