#!/usr/bin/python
#coding:utf-8

import tensorflow as tf

x = tf.placeholder("float", [None, 784])

"""
一个Variable代表一个可修改的张量
它们可以用于计算输入值，也可以在计算中被修改。
在这里，我们都用全为零的张量来初始化W和b。
因为我们要学习W和b的值，它们的初值可以随意设置。
"""

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

"""
我们赋予tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。

注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。b的形状是[10]，所以我们可以直接把它加到输出上面。

现在，我们可以实现我们的模型啦。只需要一行代码！
"""

y = tf.nn.softmax(tf.matmul(x,W) + b)
