#!/usr/bin/env python3

def func(w1, w2):
    return 2 * w1**2 + 3 *w1 *w2

import tensorflow as tf
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = func(w1, w2)

print(tape.gradient(z, [w1, w2]))


