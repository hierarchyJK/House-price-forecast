# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-01-25 12:38:26
@month:一月
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np
"""数据可视化"""
sns.set(style='whitegrid', palette='dark')
df0 = pd.read_csv("F:\\tensorflow-101-master\\notebook-examples\\chapter-4\\data0.csv",names=['square','price'])
sns.lmplot("square","price",df0,fit_reg=True)####lmplot专门用于线性关系可视化，适用于回归模型
df1 = pd.read_csv("F:\\tensorflow-101-master\\notebook-examples\\chapter-4\\data1.csv",names=['square','bedroom','price'])
fig = plt.figure()
"""数据归一化"""
def normalize_feature(df):
    return df.apply(lambda column:(column - column.mean()) / column.std())

df = normalize_feature(df1)
ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedroom')
ax.set_zlabel('price')
ax.scatter(df['square'],df['bedroom'],df['price'],c = df['price'],cmap='Reds') # c参数代表根据price颜色Reds渐变
plt.show()

"""数据处理，在第一列加ones列"""
ones = pd.DataFrame({'ones':np.ones(len(df))})
df = pd.concat([ones,df],axis=1)###根据列合并数据
##取表格数据：X_data 和y_data
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df),1)
print(df.head())
print(X_data.shape,y_data.shape)
print(df.info())

"""创建线性回归模型"""
def LinearRegression_model():
    alpha = 0.01 # 学习率alpha
    epoch = 500 # 训练全量数据的轮数

    #输入数据
    X = tf.placeholder(tf.float32, shape = X_data.shape)
    y = tf.placeholder(tf.float32, shape = y_data.shape)

    #初始化权重
    W = tf.get_variable(name="weights", shape=(X_data.shape[1],1),initializer=tf.constant_initializer())

    #假设函数h(x) = w0*x0 + w1*x1 + w2*x2 ,其中x0恒为1，预测这y_pred 形状(47,1)
    y_pred = tf.matmul(X,W)

    #损失函数采用最小二乘法，y_pred-y是形如[47, 1]的向量
    #tf.matmul(a,b,transpose_a=True)  表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
    loss_op = 1/(2 * len(X_data)) * tf.matmul((y_pred - y),(y_pred - y),transpose_a=True)

    #随机梯度下降优化器opt
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    #单轮训练操作
    train_op = opt.minimize(loss_op)

    #创建回话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #开始训练，因为训练集较小，所以采用梯度下降优化算法，每次使用全量数据进行训练
        for e in range(1, epoch + 1):
            sess.run(train_op, feed_dict={X:X_data, y:y_data})
            if e % 10 ==0:
                loss, w = sess.run([loss_op, W],feed_dict={X:X_data, y:y_data})
                log_str = "Epoch %d \t Loss=%.4g \t Model:y = %.4gx1 + %.4gx2 + %.4g"
                print(log_str % (e, loss, w[1], w[2], w[0]))
LinearRegression_model()

