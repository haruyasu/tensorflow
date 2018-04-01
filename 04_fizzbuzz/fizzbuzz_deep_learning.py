import numpy as np
import tensorflow as tf

def tensorflow():
    NUM_DIGITS = 10

    def binary_encode(i, num_digits):
        return np.array([i >> d & 1 for d in range(num_digits)])

    def fizz_buzz_encode(i):
        if i % 15 == 0:
            res = np.array([0, 0, 0, 1])
        elif i % 5 == 0:
            res = np.array([0, 0, 1, 0])
        elif i % 3 == 0:
            res = np.array([0, 1, 0, 0])
        else:
            res = np.array([1, 0, 0, 0])
        return res

    trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h, w_o):
        h = tf.nn.relu(tf.matmul(X, w_h))
        return tf.matmul(h, w_o)

    X = tf.placeholder("float", [None, NUM_DIGITS])
    Y = tf.placeholder("float", [None, 4])

    NUM_HIDDEN = 100

    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, 4])

    py_x = model(X, w_h, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    predict_op = tf.argmax(py_x, 1)

    def fizz_buzz(i, prediction):
        return [str(i), "Fizz", "Buzz", "FizzBuzz"][prediction]

    BATCH_SIZE = 128

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for epoch in range(10000):
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start: end]})

            if epoch % 100 == 0:
                print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))

        numbers = np.arange(1, 101)
        teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(fizz_buzz)(numbers, teY)

        print(output)
        return output

def fizzbuzz(n):
    if n % 3 == 0 and n % 5 == 0:
        res = "FizzBuzz"
    elif n % 3 == 0:
        res = "Fizz"
    elif n % 5 == 0:
        res = "BUzz"
    else:
        res = str(n)
    return res

if __name__ == "__main__":
    tf_output = tensorflow()
    c = 0
    for i in range(1, 101):
        result = tf_output[i - 1]
        answer = fizzbuzz(i)
        if result == answer:
            c += 1
    print(str(c) + "%")
