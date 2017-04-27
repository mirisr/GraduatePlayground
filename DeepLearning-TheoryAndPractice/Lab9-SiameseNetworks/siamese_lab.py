#%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from PIL import Image
import random
import itertools
import time
import math
import pdb
import sys
#----------------------FUNCTIONS---------------------
#diffImages: 16552692
#sameImages: 484514
#epochs = 2
#batch_size = 1
#13233
epochs = 15
batch_size = 100
#13233
def createData(numOfFiles=10000, numOfTrainingSamples=10000, numOfTestingSamples = 2000, size=25):
    files = open( './list.txt' ).readlines()
    #Reset them to all the files
    #numOfFiles = len(files)
    data = np.zeros(( numOfFiles, size, size ))
    labels = np.zeros(( numOfFiles, 1 ))

    # a little hash map mapping subjects to IDs
    ids = {}
    scnt = 0
    # sames = {} => label: [index1, index2,...,indexn]
    sames = {}

    # load in all of our images
    index = 0
    for fn in files:
        #print "fn:", fn
        subject = fn.split('/')[3]
        #print "Subject:", subject
        if not ids.has_key( subject ):
            ids[ subject ] = scnt
            scnt += 1
        label = ids[ subject ]
        #print "label:", label
        img = Image.open( fn.rstrip() )
        img = img.resize((size, size), Image.ANTIALIAS)
        data[ index, :, : ] = np.array( img )
        
        #print "index:", index
        labels[ index ] = label
        if not sames.has_key(label):
            sames[label] = []
       
        sames[label].append(index)

        index += 1
        if index == numOfFiles:
            break


    #------------------------SPLITTING DATA-----------------------------
    trainIndexes = []
    testIndexes = []
    sameImgs = []
    diffImgs = []
    singleIndexes = []
    #for s in sames:
    #    print s, ": ", sames[s]

    for s in sames:
        if len(sames[s]) > 1:
            #print s, ": ", sames[s]
            a = sames[s]
            b = sames[s]
            comb = list(itertools.product(a,b))

            sameImgs.extend(comb)
            #print s, ": ", comb
        else:
            #print s, ": ", sames[s]
            singleIndexes.extend(sames[s])

    #print singleIndexes
    a = singleIndexes
    b = singleIndexes
    diffImgs = list(itertools.product(a,b))
    diffImgs = [i for i in diffImgs if i[0] != i[1]]
    sameImgs = [i for i in sameImgs if i[0] != i[1]]

    #print "diffImages:" , len(diffImgs)
    #print "sameImages:" , len(sameImgs)

    #Shuffle the ids combinations
    random.shuffle(sameImgs)
    random.shuffle(diffImgs)

    #Grab the id combos (the amoung specified)
    trainSameImgs_indexes = sameImgs[:numOfTrainingSamples]
    testSameImgs_indexes = sameImgs[numOfTrainingSamples: numOfTrainingSamples + numOfTestingSamples]

    trainDiffImgs_indexes = diffImgs[:numOfTrainingSamples]
    testDiffImgs_indexes = diffImgs[numOfTrainingSamples: numOfTrainingSamples + numOfTestingSamples]
    
    #Now to actually put the images into data structures (instead of just indexes)
    trainSameImgs = []
    trainDiffImgs = []

    testSameImgs = []
    testDiffImgs = []

    for s in trainSameImgs_indexes:
        index1 = s[0]
        index2 = s[1]
        trainSameImgs.append((data[index1], data[index2]))


    for s in trainDiffImgs_indexes:
        index1 = s[0]
        index2 = s[1]
        trainDiffImgs.append((data[index1], data[index2]))

    for s in testSameImgs_indexes:
        index1 = s[0]
        index2 = s[1]
        testSameImgs.append((data[index1], data[index2]))


    for s in testDiffImgs_indexes:
        index1 = s[0]
        index2 = s[1]
        testDiffImgs.append((data[index1], data[index2]))


    #mix the test data with diff and same images
    

    return trainSameImgs, trainDiffImgs, testSameImgs, testDiffImgs

#next_batch(same, s, e, trainSameImgs, trainDiffImgs)
def next_batch(same, s,e, trainSameImgs, trainDiffImgs):
    #print "trainSameImgs:", trainSameImgs[0]
    size = batch_size
    #print "size: ", size
    batchData1 = np.zeros(( size, 25, 25 ))
    batchData2 = np.zeros(( size, 25, 25 ))

    if same:
        trainImgs = trainSameImgs[s:e]
        batchLabels = np.zeros((size, 1 )) # 0 for same
    else:
        trainImgs = trainDiffImgs[s:e]
        batchLabels = np.ones((size, 1 )) # 1 for imposter
    
    for i in xrange(len(trainImgs)):
            batchData1[ i ] = trainImgs[i][0]
            batchData2[ i ] = trainImgs[i][1]

    return batchData1,batchData2,batchLabels

def compute_accuracy(prediction,labels):
    #print "prediction shape:",len( prediction)
    #print "labels", labels
    #print "_____________"
    #print "predictions:", prediction
    
    #return labels[prediction.ravel() < 0.5].mean()
    #print "label:", labels[0][0]
    #print "prediction:", prediction
    if labels[0][0] == 1:
        #print "label is 1 imposter"
        if prediction < 2:
            return 1
        else: 
            return 0
    else:
        #print "label is 0"
        if prediction > 2:
            return 1
        else:
            return 0
    return 0

    
def contrastiveLoss(y,energy,m=4):
    tmp= y * tf.square(energy)
    tmp2 = (1-y) * tf.square(tf.maximum((m - energy),0))
    return tf.reduce_sum(tmp +tmp2) #/batch_size/2

def linear( in_var, output_size, name="linear", stddev=0.02, bias_val=0.0 ):
    shape = in_var.get_shape().as_list()
    
    with tf.variable_scope( name):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
       
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b


def conv2d( in_var, output_dim, ksize = 3, stride = 2, name="conv2d" ):
    # filter width/height
    # x,y strides

    with tf.variable_scope( name):
        W = tf.get_variable( "W", [ksize, ksize, in_var.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )
        #b = tf.get_variable( "b", [output_dim], initializer=tf.constant_initializer(0.0) )

        conv = tf.nn.conv2d( in_var, W, strides=[1, stride, stride, 1], padding='SAME' )
        #conv = tf.reshape( tf.nn.bias_add( conv, b ), conv.get_shape() )

        return conv  

def maxpool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')  


def bn(in_var, output_dim, ksize = 3, name = "bn"):

    mean, variance = tf.nn.moments(in_var, axes=[0,1,2])
    #beta = tf.Variable(tf.zeros([output_dim]), name="beta")
    #gamma = tf.Variable(tf.truncated_normal([output_dim], stddev=0.1), name="gamma")
    
    with tf.variable_scope(name):
        beta = tf.get_variable( "beta", initializer=tf.zeros([output_dim]) )

        gamma = tf.get_variable( "gamma", [output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.02) )


    batch_norm = tf.nn.batch_norm_with_global_normalization(
        in_var, mean, variance, beta, gamma, 0.001,
        scale_after_normalization=True)

    return batch_norm

def block(in_var, output_dim, stride):

    with tf.variable_scope( "block_conv_1" ):
        conv_1 = conv2d(in_var, output_dim, 3, stride)
        r_bn_1 = bn(conv_1, output_dim)
        r_activation_1 = tf.nn.relu(r_bn_1) 

    with tf.variable_scope( "block_conv_2" ):
        conv_2 = conv2d(r_activation_1, output_dim, 3, stride)
        r_bn_2 = bn(conv_2, output_dim)

    input_dim = int(in_var.get_shape()[-1])
    #print "input_dim", input_dim
    #print "output_dim", output_dim
    if input_dim != output_dim:
        in_var = tf.pad(in_var, [[0,0], [0,0], [0,0], [0, output_dim - input_dim]])

    return tf.nn.relu(in_var + r_bn_2)


#Write Identify Model function HERE
#---------------------
def resNet( imgs ):
    #reshaping to make compatible with convolution
    #print "bathc_size:", batch_size
    imgs = tf.reshape( imgs, [ batch_size, 25, 25, 1 ] )

    with tf.variable_scope( "r_conv_1" ):
        r_conv_1 = conv2d(imgs, 64, 7, 2)
        r_bn_1 = bn(r_conv_1, 64)
        r_activation_1 = tf.nn.relu(r_bn_1) 

    with tf.variable_scope( "r_maxpool_1" ):
        r_maxpool_1 = maxpool(r_activation_1)

    conv = r_maxpool_1
    for i in xrange(2):#3
        with tf.variable_scope( "r_conv_64s_%d" % (i) ):
            conv = block(conv, 64, 1)

    for i in xrange(3):#4
        with tf.variable_scope( "r_conv_128s_%d" % (i) ):
            conv = block(conv, 128, 1)
    
    for i in xrange(3):#5
        with tf.variable_scope( "r_conv_256s_%d" % (i) ):
        
            conv = block(conv, 256, 1)

    for i in xrange(2):#3
        with tf.variable_scope( "r_conv_512s_%d" % (i) ):
            conv = block(conv, 512, 1)

    with tf.variable_scope('fc'):
        avgpool = tf.reduce_mean(conv, [1, 2])
        fc_linear = linear(avgpool, 10)
        fc = tf.nn.softmax(fc_linear)

    return fc


#Write Computation model here
#---------------------



data_one = tf.placeholder( tf.float32, shape=[None, 25, 25], name="data_one" )
data_two = tf.placeholder( tf.float32, shape=[None, 25, 25], name="data_two" )
labels = tf.placeholder( tf.float32, shape=[None, 1], name="labels" )


print "...Setting up Computation Graph"
with tf.name_scope( "Siamese") as scope:
    #print "About to do resNet1"
    resNet1= resNet(data_one)
    tf.get_variable_scope().reuse_variables()
    #print "About to do resNet2"
    resNet2 = resNet(data_two)
    #print resNet1.get_shape()
    #print resNet2.get_shape()


energy = tf.reduce_sum(tf.abs(tf.sub(resNet1,resNet2)))
loss = contrastiveLoss(labels,energy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(loss)

print "...DONE"

#----------------------CREATE DATA  -----------------------
print "...loading data"
#trainImgs, trainLabs, testImgs, testLabs  = createData()
#trainSameImgs, trainSameLabs, trainDiffImgs, trainDiffLabs, testImgs, testLabs = createData()
trainSameImgs, trainDiffImgs, testSameImgs, testDiffImgs = createData()

print "...DONE"

#-------------BASELINE---------------- 
'''
correct = 0
incorrect = 0
for pair in testData:
    
    guess = random.randint(0,1)

    label0 = labels[pair[0]]
    label1 = labels[pair[1]]

    if label0 == label1 and guess == 1:
        correct+=1
    elif label0 != label1 and guess == 0: 
        correct += 1
    else:
        incorrect+=1

print "Accuracy:", correct/float(correct+incorrect)
'''

#--------END BASELINE-----------------


#Write Session Here
#----------------------
sess = tf.Session()
sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

print "Staring Epochs..."
for i in range( epochs ):
    avg_loss = 0.
    avg_acc = 0.
    #print "batch_size:", batch_size
    #print "len(trainSameIm *2", (len(trainSameImgs) * 2 )
    total_batch = int((len(trainSameImgs) * 2 )/batch_size)
    #print "total_batch:", total_batch
    start_time = time.time()
    # Loop over all batches
    same = False
    every2 = -1
    for j in range(total_batch):
        if same:
            same = False
        else:
            same = True
            every2 += 1

        s  = every2 * batch_size
        e = (every2+1) *batch_size
        # Fit training using batch data
        batchData1, batchData2, batchLabels = next_batch(same, s, e, trainSameImgs, trainDiffImgs)
       # print "Running through graph: batch: ", j
        _,loss_value, predict = sess.run([optimizer, loss, energy], feed_dict={data_one:batchData1, data_two:batchData2, labels:batchLabels})
        
        tr_acc = compute_accuracy(predict , batchLabels)
        avg_acc += tr_acc*100
        avg_loss += loss_value

    duration = time.time() - start_time
    print('epoch %d  time: %f loss %0.5f accuracy %0.2f' %(i,duration,avg_loss/(total_batch),float(avg_acc)/total_batch))
    sys.stdout.flush()



print "-----------test model----------"
# Test model
#testImgs, testLabs
avg_acc = 0.
#print "total test batch/ len(testIms): ", len(testImgs)

#print "test imgs:", testImgs[0]
index = -1
total_batch = int((len(testSameImgs) * 2 )/batch_size)
#print "total_batch:", total_batch
start_time = time.time()
# Loop over all batches
same = False
every2 = -1

for j in range(total_batch):
    if same:
        same = False
    else:
        same = True
        every2 += 1

    s  = every2 * batch_size
    e = (every2+1) *batch_size
    # Fit training using batch data
    batchData1, batchData2, batchLabels = next_batch(same, s, e, testSameImgs, testDiffImgs)
   # print "Running through graph: batch: ", j
    predict = sess.run(energy, feed_dict={data_one:batchData1, data_two:batchData2, labels:batchLabels})
    #print "----------"
    #print "same:", same
    #print "energy:", predict
    te_acc = compute_accuracy(predict , batchLabels)
    #print "acc:", te_acc*100
    avg_acc += te_acc*100
    #print "----------"
    #print "avg_acc:", avg_acc
    #print "total_batch:", total_batch

print "Test Accuracy: %0.2f" % (float(avg_acc)/total_batch)





