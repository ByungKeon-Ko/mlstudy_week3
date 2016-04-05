import tensorflow as tf
import cPickle
import numpy as np
import Image
import random

print "week4_1.py start!!"

# --- Image Loading function from CIFAR webpage Guide --------------------- #
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# --- Constant ------------------------------------------------------------ #
red_index = 0
green_index = 1
blue_index = 2

im_len = 32         # 32 x 32 pixels image
im_size = 1024      # 32 x 32 pixels image
num_classes = 10
patch_size = 20
nLearning = 100
LEARNING_RATE = 0.001

TEST_MODE = 0       # '0' : Learning Mode, '1' : Test Mode w/o Learning steps

nImage_inTrBatch = 10000*5       # the number of Images in one Batch
nImage_inTeBatch = 10000

nMiniBatch = nImage_inTrBatch / patch_size

ofile_trloss = "train_loss.txt"
ofile_trcorr = "train_correct.txt"
ofile_teloss = "test_loss.txt"
ofile_tecorr = "test_correct.txt"

f_trloss = open(ofile_trloss, "w")
f_trcorr = open(ofile_trcorr, "w")
f_teloss = open(ofile_teloss, "w")
f_tecorr = open(ofile_tecorr, "w")

# --- Image Load ------------------------------------------------------------ #
batchfile_1 = "./cifar_dataset/cifar-10-batches-py/data_batch_1"
batchfile_2 = "./cifar_dataset/cifar-10-batches-py/data_batch_2"
batchfile_3 = "./cifar_dataset/cifar-10-batches-py/data_batch_3"
batchfile_4 = "./cifar_dataset/cifar-10-batches-py/data_batch_4"
batchfile_5 = "./cifar_dataset/cifar-10-batches-py/data_batch_5"
testbatch = "./cifar_dataset/cifar-10-batches-py/test_batch"

dict_image_train = []
dict_label_train = []
dict_image_test = []
dict_label_test = []

batch_1 = unpickle(batchfile_1)
batch_2 = unpickle(batchfile_2)
batch_3 = unpickle(batchfile_3)
batch_4 = unpickle(batchfile_4)
batch_5 = unpickle(batchfile_5)

dict_image_train.extend ( batch_1['data'] )
dict_label_train.extend ( batch_1['labels'] )
dict_image_train.extend ( batch_2['data'] )
dict_label_train.extend ( batch_2['labels'] )
dict_image_train.extend ( batch_3['data'] )
dict_label_train.extend ( batch_3['labels'] )
dict_image_train.extend ( batch_4['data'] )
dict_label_train.extend ( batch_4['labels'] )
dict_image_train.extend ( batch_5['data'] )
dict_label_train.extend ( batch_5['labels'] )

batch = unpickle(testbatch)
dict_image_test.extend ( batch['data'] )
dict_label_test.extend ( batch['labels'] )

# --- Chaning Color Space ------------------------------------------------------------ #

def color_conversion(nImage_inBatch, dict_image) :
    im_matrix = np.reshape(dict_image, [nImage_inBatch, 3, im_len, im_len ] )
    im_matrix = np.transpose(im_matrix, [0, 2,3, 1] )
    
    bw_image = np.zeros( (nImage_inBatch, im_len,im_len) ).astype('float64')
    
    for i in range(0,nImage_inBatch) :
        tmp_array = Image.fromarray(im_matrix[i])
        tmp_array = tmp_array.convert('YCbCr')
        bw_image[i] = np.asarray(tmp_array)[:,:,0]  # Choos only Y value so it changes to black and white image
        if i==0 :
            tmp_array.show()
    
    bw_image = np.reshape(bw_image, [nImage_inBatch, im_size] )
    bw_image = bw_image/255
    return bw_image

train_image = color_conversion(nImage_inTrBatch, dict_image_train )
test_image  = color_conversion(nImage_inTeBatch, dict_image_test )

print "Image Converting Done!!"
# tmp_array.show()

# --- Changing Labels to one-hote encoding ------------------------------------------- #
def change_to_onehot(nImage_inBatch, dict_label) :
    dict_label = dict_label[0:nImage_inBatch]
    label_onehot = np.zeros([nImage_inBatch, num_classes] ).astype('uint8')
    for i in range(0, nImage_inBatch):
        if dict_label[i]==0 :
            label_onehot[i] = (0,0,0,0,0,0,0,0,0,1)
        elif dict_label[i]==1 :
            label_onehot[i] = (0,0,0,0,0,0,0,0,1,0)
        elif dict_label[i]==2 :
            label_onehot[i] = (0,0,0,0,0,0,0,1,0,0)
        elif dict_label[i]==3 :
            label_onehot[i] = (0,0,0,0,0,0,1,0,0,0)
        elif dict_label[i]==4 :
            label_onehot[i] = (0,0,0,0,0,1,0,0,0,0)
        elif dict_label[i]==5 :
            label_onehot[i] = (0,0,0,0,1,0,0,0,0,0)
        elif dict_label[i]==6 :
            label_onehot[i] = (0,0,0,1,0,0,0,0,0,0)
        elif dict_label[i]==7 :
            label_onehot[i] = (0,0,1,0,0,0,0,0,0,0)
        elif dict_label[i]==8 :
            label_onehot[i] = (0,1,0,0,0,0,0,0,0,0)
        elif dict_label[i]==9 :
            label_onehot[i] = (1,0,0,0,0,0,0,0,0,0)
    return label_onehot

train_label = change_to_onehot(nImage_inTrBatch, dict_label_train)
test_label  = change_to_onehot(nImage_inTeBatch, dict_label_test)

# Garbage memory release
batch_1 = []
batch_2 = []
batch_3 = []
batch_4 = []
batch_5 = []
batch = []
dict_image_train = []
dict_image_test  = []
dict_label_train = []
dict_label_test  = []

# --- Designing Network --------------------------------------------- #
x = tf.placeholder( tf.float32, [None, im_size], name = 'x' )

# W = tf.Variable( tf.ones([im_size, num_classes])/10 )
# b = tf.Variable( tf.ones([num_classes])/10 )
# W = tf.Variable( tf.random_normal([im_size, num_classes], mean=0.0, stddev=1.0, dtype=tf.float32 ) )
# b = tf.Variable( tf.random_normal([num_classes], mean=0.0, stddev=1.0, dtype=tf.float32 ) )
W = tf.Variable( tf.zeros([im_size, num_classes]) )
b = tf.Variable( tf.zeros([num_classes]) )
y = tf.nn.softmax(tf.matmul(x,W) +b )

# --- Cost : softmax, cross-entropy ------------------------------------------------ #
y_ = tf.placeholder(tf.float32, [None ,num_classes], name = 'y_' )
cross_entropy = -tf.reduce_mean(y_*tf.log(y))

# --- Training    ------------------------------------------------ #
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

if TEST_MODE==0 :
    init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") )

saver = tf.train.Saver()

sess = tf.InteractiveSession()

if TEST_MODE==0 :
    sess.run(init)
    print " Training Start !!!!!!!! "

    epoch = 0
    cnt_minibatch = 0     # ex) if patch_size = 100, nImage_inTrBatch=4000, cnt_minibatch should be 0~39
    index_array = range(0,nImage_inTrBatch)
    max_index = nImage_inTrBatch -1
    loss_sum = 0

    while epoch < nLearning :
    
        # --- Mini-batch ------------------------------------- #
        if cnt_minibatch == nMiniBatch -1 :
            epoch = epoch+1
            cnt_minibatch = 0
            index_array = range(0,nImage_inTrBatch)
            max_index = nImage_inTrBatch -1
        else :
            cnt_minibatch = cnt_minibatch+1
    
        x_batch = []
        y_batch = []
        if (patch_size==1) :
            x_batch = train_image
            y_batch = train_label
        else :
            for i in range(patch_size) :
                rand_index = index_array.pop( random.randint(0,max_index) )
                max_index = max_index -1
                x_batch.append( train_image[rand_index] )
                y_batch.append( train_label[rand_index] )
        # --------------------------------------------------- #

        sess.run(train_step, feed_dict={x:x_batch , y_:y_batch } )
        loss_sum = loss_sum + sess.run(cross_entropy, feed_dict={x:x_batch, y_:y_batch} )

        if cnt_minibatch==0 :
            tr_loss = loss_sum / nMiniBatch
            loss_sum = 0
            tr_corr = sess.run(accuracy,      feed_dict={x:x_batch, y_:y_batch} )
            te_loss = sess.run(cross_entropy, feed_dict={x:test_image, y_:test_label})
            te_corr = sess.run(accuracy,      feed_dict={x:test_image, y_:test_label})
            f_trloss.write("%f\n" % tr_loss)
            f_trcorr.write("%f\n" % tr_corr)
            f_teloss.write("%f\n" % te_loss)
            f_tecorr.write("%f\n" % te_corr)

        # if cnt_minibatch % 10000 == 0 :
        #     print "cnt_minibatch = ", cnt_minibatch

        if (epoch%10 == 0) & (cnt_minibatch==0) :
            print "epoch = ", epoch
            print "Train Cross Entropy = ", tr_loss
            print "Train correct rate  = ", tr_corr
            print "Test Cross Entropy = ",  te_loss
            print "Test correct rate  = ",  te_corr
            # print "W= ", W.eval()[0]
            # print "B= ", b.eval()

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print "Model saved in file: ", save_path
    
    print "W= ", W.eval()
    print "B= ", b.eval()
    f_trloss.close()
    f_trcorr.close()
    f_teloss.close()
    f_tecorr.close()
    # sess.close()

else :
    saver.restore(sess, "/tmp/model.ckpt")
    print "Model restored, Testing Start!! "

    print "Test Cross Entropy = ", sess.run(cross_entropy, feed_dict={x:test_image, y_:test_label} )
    print "Test correct rate  = ", sess.run(accuracy,      feed_dict={x:test_image, y_:test_label} )
    

