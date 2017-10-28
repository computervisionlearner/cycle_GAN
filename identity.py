#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:54:16 2017

@author: no1
"""
import tensorflow as tf
x = tf.Variable(1.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = x
    z=tf.identity(x,name='x')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        print(sess.run(z))
        



        