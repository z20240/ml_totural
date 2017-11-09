#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 10)) # 随机输入

print("x_data:", x_data, "\n")

y_data = np.dot([0.100, 0.200], x_data) + 0.300

print("y_data:", y_data, "\n")

# 构造一个线性模型
# 
b = tf.Variable(tf.zeros([1]))

print("b:", b, "\n")

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

print("W:", W, "\n")

y = tf.matmul(W, x_data) + b

print("[new] y:", y, "\n")

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))

print("loss:", loss, "\n")

optimizer = tf.train.GradientDescentOptimizer(0.5)

print("optimizer:", optimizer, "\n")

train = optimizer.minimize(loss)

print("train:", train, "\n")

# 初始化变量
init = tf.initialize_all_variables()


print("init:", init, "\n")

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

print("===  SESS RUN ===", "\n")

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
