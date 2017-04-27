
import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize, imsave
from PIL import Image
import sys
# Constant to put more emphasis on content loss.
BETA = 10000
# Constant to put more emphasis on style loss.
ALPHA = 1

def get_gram_matrix(F, N, M):
	F_ = tf.reshape(F, (M,N))
	F_T = tf.transpose(F_)
	return tf.matmul(F_T, F_)


def get_style_loss_for_layer(orig_layer, gen_layer):
	N = orig_layer.shape[3]
	M = orig_layer.shape[1] * orig_layer.shape[2]

	# A - Style representation of original image at layer 
	A = get_gram_matrix(orig_layer, N, M)

	# G - Style representation of generated image at layer
	G = get_gram_matrix(gen_layer, N, M)

	return (1.0 / (2.0 * N**2 * M**2)) * tf.nn.l2_loss(G - A)



sess = tf.Session()

opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'style.png', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'content_iris.jpg', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

init_img = imread( 'white_noise.png', mode='RGB' )
init_img = imresize( init_img, (224, 224) )
init_img = np.reshape( init_img, [1,224,224,3] )


layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]


#These are our activations
content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

#print "model vgg:", 
#print "content_acts: ", len(content_acts), content_acts[0].shape
#print "style_acts: ", len(style_acts), style_acts[0].shape
#
# --- construct your cost function here
#

content_layer = content_acts[layers.index("conv4_2")]

content_loss = tf.nn.l2_loss(vgg.conv4_2 - content_layer)



style_layers = []
style_layers.append(style_acts[layers.index("conv1_1")])
style_layers.append(style_acts[layers.index("conv2_1")])
style_layers.append(style_acts[layers.index("conv3_1")])
style_layers.append(style_acts[layers.index("conv4_1")])
style_layers.append(style_acts[layers.index("conv5_1")])

style_loss_per_layer = []

style_loss_per_layer.append(get_style_loss_for_layer(style_layers[0], vgg.conv1_1))
style_loss_per_layer.append(get_style_loss_for_layer(style_layers[1], vgg.conv2_1))
style_loss_per_layer.append(get_style_loss_for_layer(style_layers[2], vgg.conv3_1))
style_loss_per_layer.append(get_style_loss_for_layer(style_layers[3], vgg.conv4_1))
style_loss_per_layer.append(get_style_loss_for_layer(style_layers[4], vgg.conv5_1))

style_loss = 0
for loss in style_loss_per_layer:
	weight = (1.0 / 5.0)
	style_loss += loss * weight

total_loss = ALPHA * content_loss + BETA * style_loss

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)

optimize = tf.train.AdamOptimizer(.1, beta1=0.5).minimize(total_loss, var_list=[opt_img] )

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
sess.run( opt_img.assign( content_img ))

# --- place your optimization loop here

for epoch in xrange(6000):
	new_image = sess.run(opt_img)
	_, loss, content, style = sess.run([optimize, total_loss, content_loss, style_loss])
	print "Epoch:", epoch, " Total Loss:", loss, "Content Loss:", content, "Style Loss:", style
	
	if epoch % 100 == 0:
		clipped_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
		sess.run( opt_img.assign( clipped_img ))


image = new_image[0]
image = np.clip(image, 0, 255).astype('uint8')
imsave("iris_starry_art.png", image)

