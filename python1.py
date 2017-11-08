import tensorflow as tf
import numpy as np

def test1():
    x_data = np.random.rand(100).astype(np.float64)
    y_data = x_data * 0.1 + 0.3


    Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    biase = tf.Variable(tf.zeros([1]))


    y = Weights*x_data + biase

    loss = tf.reduce_mean(tf.square(y-y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)



    init = tf.initialize_all_variables()


    with tf.Session() as sess:
        sess.run(init)
        for step in range(300):
            sess.run(train)
            if step % 20 == 0:
                print(step,sess.run(Weights),sess.run(biase))

# test1()