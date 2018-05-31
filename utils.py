import numpy as np, pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def get_variable(shape, initial=None):
    if initial is None:
        initial = tf.random_normal(shape)
        return tf.Variable(initial)
    else:
        return tf.Variable(initial)
        

def HZ_reg(weights):
    d1, d2 = weights.shape
    d1, d2 = int(d1), int(d2)
    n_chan = d2//d1
    weights_rolled = tf.manip.roll(tf.reshape(weights,[d1,d1,n_chan]), shift=[-1], axis=[0])
    result = tf.reduce_mean(tf.abs(tf.reshape(weights,[d1,d1,n_chan])-weights_rolled))
    return result

def DZ_reg(weights):
    d1, d2 = weights.shape
    d1, d2 = int(d1), int(d2)
    n_chan = d2//d1
    weights_rolled = tf.manip.roll(tf.reshape(weights,[d1,d1,n_chan]), shift=[-1,-1], axis=[0,1])
    result = tf.reduce_mean(tf.abs(tf.reshape(weights,[d1,d1,n_chan])-weights_rolled))
    return result

def regularize(loss, weights, reg_kind=None, reg_value=None):
    if reg_kind is None:
        return loss
    elif reg_kind=='L1':
        print("Adding L1 regularization")
        return loss + reg_value*tf.reduce_mean(tf.abs(weights))
    elif reg_kind=='HZ':    
        print("Adding HZ regularization")
        return loss + reg_value*HZ_reg(weights)
    elif reg_kind=='DZ':    
        print("Adding DZ regularization")
        return loss + reg_value*DZ_reg(weights)
    elif reg_kind=='DZ+L1':    
        print("Adding DZ+L1 regularization")
        return loss + reg_value[0]*DZ_reg(weights)+reg_value[1]*tf.reduce_mean(tf.abs(weights))
    else:
        raise ValueError("Invalid value for parameter: reg_kind")

def visualize_general_weights(weights, num_channels, num_digits=6, pool=4, suptitle=None):
    weights = np.abs(weights.reshape(int(28/pool)**2, int(28/pool)**2, num_channels)) # we care about magnitude
    fig, axes = plt.subplots(int((num_digits+2)/3), 3, figsize=[9, 3*int((num_digits+2)/3)])
    for i in range(num_digits):
        axes[int(i/3), i%3].imshow(weights[:, :, i], cmap='gray', vmin=np.min(weights), vmax=np.max(weights))          
    if not(suptitle is None):
        plt.suptitle(suptitle)
    
def convolutional(mnist, pool=4, num_channels=10, verbose=True, batches=1500, print_every=100,
                  return_weights=False,reg_kind=None, reg_value=None, filter_size=3):

    d = int(784/pool**2)
    x = tf.placeholder(tf.float32, [None, 784])

    #hidden layer
    W_conv = get_variable([filter_size, filter_size, 1, num_channels])
    b_conv = get_variable([num_channels])
    #h_fc1 = tf.sigmoid(tf.matmul(x, W_fc1) + b_fc1)

    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    x_reshaped = tf.nn.max_pool(x_reshaped, [1, pool, pool,1], [1, pool, pool, 1], padding='SAME')
    #x_reshaped = tf.reshape(x_reshaped, [-1, d])

    h = tf.nn.conv2d(x_reshaped, W_conv, [1, 1, 1, 1], "SAME")
    h = tf.nn.relu(tf.nn.bias_add(h, b_conv))

    #output layer
    W_fc2 = get_variable([d*num_channels, 10])
    b_fc2 = get_variable([10])
    y = tf.matmul(tf.reshape(h, [-1, d*num_channels]), W_fc2) + b_fc2

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    loss = regularize(loss, W_fc2, reg_kind, reg_value)
    train_step = tf.train.AdamOptimizer(0.003).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(batches):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if i%print_every==0:
                accuracy_train = accuracy.eval(feed_dict={x: mnist.train.images[:1000], y_: mnist.train.labels[:1000]})
                accuracy_valid = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                if verbose:
                    print('Train acc:',accuracy_train, 'Valid acc:',accuracy_valid)
                
        return_values = [accuracy_train, accuracy_valid]
        if return_weights:
            return_values.append(sess.run(W_conv))
    
    return return_values

def fully_connected(mnist, pool=4, num_channels=10, verbose=True, batches=1500, print_every=100,
                    return_weights=False, reg_kind=None, reg_value=None):
    
    d = int(784/pool**2)
    x = tf.placeholder(tf.float32, [None, 784])


    #hidden layer
    W_fc1 = get_variable([d, d*num_channels])
    b_fc1 = get_variable([d*num_channels])
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    x_reshaped = tf.nn.max_pool(x_reshaped, [1, pool, pool,1], [1, pool, pool, 1], padding='SAME')
    x_reshaped = tf.reshape(x_reshaped, [-1, d])
    h_fc1 = tf.nn.relu(tf.matmul(x_reshaped, W_fc1) + b_fc1)

    #output layer
    W_fc2 = get_variable([d*num_channels, 10])
    # W_fc2 = weight_variable([7840, 10])
    b_fc2 = get_variable([10])
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    loss = regularize(loss, W_fc1, reg_kind, reg_value)
    train_step = tf.train.AdamOptimizer(0.003).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(batches):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if i%print_every==0:
                accuracy_train = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})
                accuracy_valid = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                if verbose:
                    print('Train acc:',accuracy_train, 'Valid acc:',accuracy_valid)
        
        return_values = [accuracy_train, accuracy_valid]
        if return_weights:
            return_values.append(sess.run(W_fc1))
        
    return return_values

def fully_connected_meta(mnist, pool=4, num_channels=10, 
                         verbose=True, batches=1500, print_every=100,
                         return_weights=False, reg_kind=None, reg_value=None):
    
    d = int(784/pool**2)
    x = tf.placeholder(tf.float32, [None, 784])

    #hidden layer
    W_fc1 = get_variable([d, d*num_channels])
    b_fc1 = get_variable([d*num_channels])
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    x_reshaped = tf.nn.max_pool(x_reshaped, [1, pool, pool,1], [1, pool, pool, 1], padding='SAME')
    x_reshaped = tf.reshape(x_reshaped, [-1, d])
    h_fc1 = tf.nn.relu(tf.matmul(x_reshaped, W_fc1) + b_fc1)
    alpha = get_variable([1])
    beta = get_variable([1])
    
    #output layer
    W_fc2 = get_variable([d*num_channels, 10])
    # W_fc2 = weight_variable([7840, 10])
    b_fc2 = get_variable([10])
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    loss = loss + alpha*tf.reduce_mean(tf.abs(weights))
    loss = loss + beta*DZ_reg(W_fc1)
    train_step = tf.train.AdamOptimizer(0.003).minimize(loss, var_list=[W_fc1, b_fc1])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(batches):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if i%print_every==0:
                accuracy_train = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})
                accuracy_valid = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                if verbose:
                    print('Train acc:',accuracy_train, 'Valid acc:',accuracy_valid)
        
        return_values = [accuracy_train, accuracy_valid]
        if return_weights:
            return_values.append(sess.run(W_fc1))
        
    return return_values