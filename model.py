
#from playground.ops import layers

import tensorflow as tf 

IMAGE_SHAPE = (48,48,1)
batch_size = 512

checkpoint_directory = '/data/face/expression/checkpoints/'

def loss(logits, labels):
	# https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
	return cross_entropy

def inference(images,keep_prob = .5):
	# https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
	def conv(x,num,kernel_size=5,stride=1,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.instance_norm):
		return tf.layers.conv2d(x,num,kernel_size,strides=(stride,stride),activation=activation,kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,padding='same')
	def fc(x,num,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
		return tf.layers.dense(x,num,activation=activation,kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)		
	def dropout(x):
		return tf.layers.dropout(x,0.25)

	base_filter = 64
	x = images # 48x48 
	
	x = conv(x,base_filter)
	x = conv(x,base_filter,stride=2) # 24x24
	x = conv(x,base_filter*2)
	x = conv(x,base_filter*2,stride=2) # 12x12
	x = conv(x,base_filter*4)
	x = conv(x,base_filter*4,stride=2) # 6x6
	x = conv(x,base_filter*8)
	x = conv(x,base_filter*8,stride=2) # 3x3

	x = tf.layers.flatten(x)
	x = fc(x,base_filter*4)
	x = dropout(x)
	x = fc(x,base_filter*4)
	x = dropout(x)
	x = fc(x,7, activation = None,kernel_regularizer=None )
	return x

def add_summaries(images,loss,learn_rate,accuracy):
    # summaries
    with tf.name_scope("images"):
        tf.summary.image("images", images)

    tf.summary.scalar("test loss", loss)
    tf.summary.scalar("test accuracy", accuracy)
    tf.summary.scalar("learn_rate", learn_rate)
    
    
    merged_summary_op = tf.summary.merge_all()
    return merged_summary_op

def get_inference_variables():
	return [v for v in tf.all_variables() if 'conv2d' in v.name or 'dense' in v.name]