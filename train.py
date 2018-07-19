from __future__ import print_function
import random
import sys
import time
import os 

import cv2
import numpy
import tensorflow as tf

import model, input_data
#from playground.face.expression import input_data

db = input_data.DataReader()


def train(learn_rate, report_steps):
    
    images = tf.placeholder(tf.float32, [None]+list(model.IMAGE_SHAPE))
    labels = tf.placeholder(tf.float32, [None,7])
    
    logits = model.inference(images, keep_prob = 1.0)

    loss = model.loss(logits,labels)
    learn_rate = tf.train.exponential_decay(learn_rate,tf.train.get_or_create_global_step(),1000,0.96,staircase=True)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a saver.
    #saver = tf.train.Saver(variables_to_save,write_version=tf.train.SaverDef.V2)#tf.all_variables())
    saver = tf.train.Saver(model.get_inference_variables(),write_version=tf.train.SaverDef.V2)

    init = tf.initialize_all_variables()


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
    
        summary_writer = tf.summary.FileWriter(model.checkpoint_directory, graph=tf.get_default_graph())
        merged_summary_op = model.add_summaries(images,loss,learn_rate,accuracy)

        # try to load trained model 
        ckpt = tf.train.get_checkpoint_state(model.checkpoint_directory)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('restored model from '+str(ckpt.model_checkpoint_path))

        def save(step):
            checkpoint_path = os.path.join(model.checkpoint_directory, 'model.ckpt')
            if not os.path.isdir(model.checkpoint_directory): os.makedirs(model.checkpoint_directory)

            saver.save(sess, checkpoint_path, global_step=step)
            print('saved model to disk')


        step = 0
        try:
            while True:
            	### Train ###
                train_patches, train_labels = db.getNextBatch(batch_size = model.batch_size)
                _, loss_value, accuracy_value  = sess.run([train_step,loss,accuracy],feed_dict={images: train_patches, labels: train_labels})
                step += 1
                if step % 50 == 0:
                    print('done with step',step,'loss =',loss_value.sum(),'accuracy =',accuracy_value)
                
                ### testing model 
                if step % 100 == 0:
                    test_patches, test_labels = db.getNextBatch(batch_size = model.batch_size, train = False)
                    
                    summary,test_loss, test_accuracy  = sess.run([merged_summary_op,loss,accuracy],feed_dict={images: test_patches, labels: test_labels})
                    print('              test loss =',test_loss.sum(),'accuracy =',test_accuracy)
                    summary_writer.add_summary(summary,step)#tf.train.get_or_create_global_step())


                ### validation
                if step % 150 == 0:
                    extra_debug = False
                    if extra_debug:
                        vlabels, vpredicted = sess.run([y_,y],feed_dict = {images: train_patches, labels:train_labels})
                        for ib,(vl, vp) in enumerate(zip(vlabels,vpredicted)[:5]):
                            print(vl,vp,'batch',train_patches[ib].min(),train_patches[ib].max())
                    

        except KeyboardInterrupt:
            pass
        finally:
            save(step)

if __name__ == "__main__":

    train(learn_rate=0.001,
          report_steps=20)



