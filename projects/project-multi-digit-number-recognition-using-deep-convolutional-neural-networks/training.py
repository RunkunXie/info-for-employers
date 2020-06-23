import os
import tensorflow as tf

from preproc import DataManager
from model import Model
from evaluator import evaluate
import pickle

def _train(main_path,
           path_to_train_tfrecord,
           path_to_validation_tfrecord,
           path_to_trainhist,
           training_log,
           restore_checkpoint,
           training_options):
    """
    training function

    :param num_train: number of training data
    :param num_val: number of validation data
    :param path_to_train_tfrecord: path to tensorflow tfrecords of training data
    :param path_to_validation_tfrecord: path to tensorflow tfrecords of validation data
    :param training_log: path to training log
    :param restore_checkpoint: path to model checkpoint
    :param training_options: training parameters
    :return: None
    """
    
    # the idea to use training option is learnt from online resources, but we design it in our own way
    batch_size = training_options['batch_size']
    learning_rate = training_options['learning_rate']
    decay_steps= training_options['decay_steps']
    decay_rate = training_options['decay_rate']
    drop_rate = training_options['drop_rate']
    stop_criterion = training_options['stop_criterion']
    max_step = training_options['max_step']
    
    with tf.Graph().as_default():
        
        # get the dataset
        dm = DataManager(main_path, dataset_type='train')
        image_batch, length_batch, digits_batch = dm.data_generator(path_to_train_tfrecord, batch_size=batch_size, is_shuffle=True)
        
        # predict length and digits
        output_length, output_digits = Model.inference(image_batch, drop_rate=drop_rate)
        
        # calculate loss
        loss = Model.loss(output_length, output_digits, length_batch, digits_batch)
        
        # update learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # the following one line of codes are referenced from online resources
        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=decay_rate,
                                                   staircase=True)
        # optimize
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        # training
        with tf.Session() as sess:

            summary_writer = tf.summary.FileWriter(training_log, sess.graph)

            # start tf session
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            saver = tf.train.Saver()
            
            print('Train the model')
            best_accuracy = 0.0
            loss_hist = []
            acc_hist = []
            
            # restore model from previous checkpoint
            if restore_checkpoint is not None:
                assert tf.train.checkpoint_exists(restore_checkpoint), '%s not found' % restore_checkpoint
                saver.restore(sess, restore_checkpoint)
                print('Restore model from: %s' % restore_checkpoint)
                
                with open(path_to_trainhist + '/losshist.pkl', 'rb') as f:
                    loss_hist = pickle.load(f)
                    f.close()
                    
                with open(path_to_trainhist + '/acchist.pkl', 'rb') as f:
                    acc_hist = pickle.load(f)
                    f.close()

            while True:
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run(
                    [train_step, loss, summary, global_step, learning_rate])
                
                # stopping criterion
                if stop_criterion == 0 or global_step_val > max_step:
                    print("Finished training")
                    break
                
                # print loss every 100 iterations
                if global_step_val % 100 == 0:
                    print("step %d, current val loss = %f" % (global_step_val, loss_val))
                    loss_hist.append(loss_val)
                    with open(path_to_trainhist + '/losshist.pkl', 'wb') as f:
                        pickle.dump(loss_hist, f)
                        f.close()
                
                # evaluate the model every 1,000 iterations
                if global_step_val % 1000 != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print("===> Evaluating val data...")
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(training_log, 'latest.ckpt'))

                accuracy = evaluate(main_path,
                                    path_to_latest_checkpoint_file,
                                    path_to_validation_tfrecord,
                                    dataset_type='validation',
                                    batch_size=128,
                                    drop_rate=0.0)
                
                print("===> accuracy = %f, best accuracy %f " % (accuracy, best_accuracy))
                acc_hist.append(accuracy)
                with open(path_to_trainhist + '/acchist.pkl', 'wb') as f:
                    pickle.dump(acc_hist, f)
                    f.close()
                
                # update the model if better performance is achieved
                if accuracy > best_accuracy:
                    checkpoint = saver.save(sess, os.path.join(training_log, 'model.ckpt'),
                                            global_step=global_step_val)
                    print("===> Model saved to file: %s" % checkpoint)
                    best_accuracy = accuracy
                    stop_criterion = training_options['stop_criterion']
                else:
                    print("===> Continue training")
                    stop_criterion -= 1
                
                print('Current stop criterion is {}'.format(stop_criterion))

            coord.request_stop()
            coord.join(threads)