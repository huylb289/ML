import tensorflow as tf


# 1. Build Graph
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)
print(result)


# 2. Build Session
# with close
sess = tf.Session()
print(sess.run(result))
sess.close()

# not with close
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
