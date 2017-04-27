

import tensorflow as tf
import numpy as np

from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
 
 #--------------------------------------------

# FUNCTIONS

def weight_variable(shape):
    initial = tf.truncated_normal( shape, stddev=0.1 )
    return tf.Variable( initial )

def bias_variable(shape):
    initial = tf.constant( 0.1, shape=shape )
    return tf.Variable(initial)

def linear( in_var, output_size, name="linear", stddev=0.02):
    shape = in_var.get_shape().as_list()
    
    with tf.variable_scope( name):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )

        return tf.matmul( in_var, W )

# END OF FUNCTIONS
# ------------------

class IrisGRUCell( RNNCell ):
 
    def __init__( self, num_units, input_size=None, activation=tanh ):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
 
    @property
    def state_size(self):
        return self._num_units
 
    @property
    def output_size(self):
        return self._num_units
    
    #inputs x_t
    #state h_t_before

    def __call__( self, inputs, state, scope=None ):
        with tf.variable_scope(scope or type(self).__name__):  # "IrisGRUCell"
            #renaming variables
            x_t = inputs
            h_t_1= state
            output_size = self._num_units

            # sigmoids
            with tf.variable_scope("Sigmoids"):
                # bias
                b_r = tf.get_variable( "b_r", [output_size], initializer=tf.constant_initializer(1.0) )
                b_z = tf.get_variable( "b_z", [output_size], initializer=tf.constant_initializer(1.0) )
                b_h = tf.get_variable( "b_h", [output_size], initializer=tf.constant_initializer(1.0) )

                with tf.variable_scope("r_x_t_linear"):
                    r_x_t_linear = linear(x_t, output_size)
                with tf.variable_scope("r_h_t_1_linear"):
                    r_h_t_1_linear = linear(h_t_1, output_size)

                r_t =  tf.nn.sigmoid( r_x_t_linear + r_h_t_1_linear + b_r )

                with tf.variable_scope("z_x_t_linear"):
                    z_x_t_linear = linear(x_t, output_size)
                with tf.variable_scope("z_h_t_1_linear"):
                    z_h_t_1_linear = linear(h_t_1, output_size)

                z_t =  tf.nn.sigmoid( z_x_t_linear + z_h_t_1_linear + b_z )
            
            # activation
            with tf.variable_scope("Activations"):

                with tf.variable_scope("h_x_t_linear"):
                    h_x_t_linear = linear(x_t, output_size)
                with tf.variable_scope("h_h_t_1_linear"):
                    h_x_t_linear = linear( r_t * h_t_1, output_size)

                tilda_h_t = tanh( h_x_t_linear + h_x_t_linear + b_h) 
            
            h_t = z_t * h_t_1 + ( 1 - z_t ) * tilda_h_t

        return h_t, h_t

#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

#batch_size = 5
#sequence_length = 6

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='firstinputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( 1, sequence_length, in_onehot )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( 1, sequence_length, targ_ph )

#print "training inputs:", inputs
# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------ That's a list


# ------------------
# YOUR COMPUTATION GRAPH HERE
with tf.name_scope( "RRNModel" ) as scope:
    # create a BasicLSTMCell
    #cell = IrisGRUCell( state_dim )
    cell = GRUCell( state_dim )

    # use it to create a MultiRNNCell
    multiRNNCell = rnn_cell.MultiRNNCell( [cell]*num_layers, state_is_tuple=True)

    # use it to create an initial_state
    # note that initial_state will be a *list* of tensors!
    initial_state = multiRNNCell.zero_state(batch_size, tf.float32)

    # call seq2seq.rnn_decoder
    with tf.variable_scope( "decoder") as scope:
        outputs, state = seq2seq.rnn_decoder(inputs, initial_state, multiRNNCell)
    # [10, 128]
    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    # logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    
    W = weight_variable([state_dim, vocab_size])
    b = bias_variable([vocab_size])

    logits = [] 
    for i in xrange(len(outputs)):
        logit = tf.matmul(outputs[i], W) + b
        logits.append(logit)

# call seq2seq.sequence_loss
with tf.name_scope( "Loss" ) as scope:
    #weights: ""List of 1D"" batch-sized float-Tensors of the same length as logits.
    const_W = [tf.ones([1, batch_size])] * sequence_length 
    loss = seq2seq.sequence_loss(logits, targets, const_W)

# create a training op using the Adam optimizer
with tf.name_scope( "Optimizer" ) as scope:
    optim = tf.train.AdamOptimizer( 0.0005, beta1=0.5 ).minimize(loss)

    final_state = state

# ------------------
# YOUR SAMPLER GRAPH HERE



# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!


s_batch_size = 1
s_sequence_length = 1

s_in_ph = tf.placeholder( tf.int32, [ s_batch_size, s_sequence_length ], name='s_inputs' )
s_in_onehot = tf.one_hot( s_in_ph, vocab_size, name="s_input_onehot" )

s_inputs = tf.split( 1, s_sequence_length, s_in_onehot )
s_inputs = [ tf.squeeze(input_, [1]) for input_ in s_inputs ]

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

with tf.name_scope( "Sampler") as scope:

    s_initial_state = multiRNNCell.zero_state(s_batch_size, tf.float32)
    # decoder_inputs: A list of 2D Tensors [batch_size x input_size]

    with tf.variable_scope( "decoder", reuse = True) as scope:
        #tf.get_variable_scope().reuse_variables()
        s_outputs, s_state = seq2seq.rnn_decoder(s_inputs, s_initial_state, multiRNNCell)

    s_logits = [] 

    for i in xrange(len(s_outputs)):
        logit = tf.matmul(s_outputs[i], W) + b
        s_logits.append(logit)

    s_probs = []
    for i in xrange(len(s_logits)):
        s_probs.append(tf.nn.softmax(s_logits[i]))
    s_final_state = s_state

# ---------- Where do I use my own GRU Cell?
#  
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.zeros((1, 1))
        x[0,0] = np.ravel( data_loader.vocab[char] ).astype('int32')

        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):

        x = np.zeros((1, 1))
        x[0,0] = np.ravel( data_loader.vocab[char] ).astype('int32')
        # plug the most recent character in...
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = s_probs
        ops.extend( list(s_final_state) )
        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()
        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

    print sample( num=60, prime="And " )

     #print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
