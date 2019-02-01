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
df0 = pd.read_csv("F:\\tensorflow-101-master\\notebook-examples\\chapter-4\\data0.csv", names=['square','price'])
sns.lmplot("square", "price", df0, fit_reg=True)####lmplot专门用于线性关系可视化，适用于回归模型,fit_reg表示是否拟合
df1 = pd.read_csv("F:\\tensorflow-101-master\\notebook-examples\\chapter-4\\data1.csv", names=['square','bedroom','price'])
fig = plt.figure()

"""数据归一化"""
def normalize_feature(df):
    return df.apply(lambda column:(column - column.mean()) / column.std())

df = normalize_feature(df1)
# 三维数据可视化
ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedroom')
ax.set_zlabel('price')
ax.scatter(df['square'], df['bedroom'], df['price'], c=df['price'], cmap='Reds') # c参数代表根据price颜色Reds渐变
plt.show()

"""数据处理，在第一列加ones列"""
ones = pd.DataFrame({'ones':np.ones(len(df))})
df = pd.concat([ones, df], axis=1)###根据列合并数据
##取表格数据：X_data 和y_data
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)
print(df.head())
print(X_data.shape, y_data.shape)
print(df.info())

"""创建线性回归模型"""
def LinearRegression_model():
    alpha = 0.01 # 学习率alpha
    epoch = 500 # 训练全量数据的轮数

    """为了是TensorBoard可视化数据流图显得不那么凌乱，我们这样使用tf.name_scope("name")使得数据流图显得更加简单易读"""
    with tf.name_scope("Input"):
        #输入数据
        X = tf.placeholder(tf.float32, shape = X_data.shape, name="X_input")
        y = tf.placeholder(tf.float32, shape = y_data.shape, name="y_input")
    with tf.name_scope("Hypothsis"):
        #初始化权重
        W = tf.get_variable(name="weights", shape=(X_data.shape[1],1),initializer=tf.constant_initializer())

        #假设函数h(x) = w0*x0 + w1*x1 + w2*x2 ,其中x0恒为1，预测这y_pred 形状(47,1)
        y_pred = tf.matmul(X, W, name="y_pred")
    with tf.name_scope("Loss"):
        #损失函数采用最小二乘法，y_pred - y是形如[47, 1]的向量
        #tf.matmul(a,b,transpose_a=True)  表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
        loss_op = 1/(2 * len(X_data)) * tf.matmul((y_pred - y),(y_pred - y), transpose_a=True)
    with tf.name_scope("Train"):
        #随机梯度下降优化器opt
        opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        #单轮训练操作
        train_op = opt.minimize(loss_op)

    #创建回话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./summary/linear-regression-0/", sess.graph)
        # 记录每轮迭代的loss值
        loss_datas = []
        #开始训练，因为训练集较小，所以采用梯度下降优化算法，每次使用全量数据进行训练
        for e in range(1, epoch + 1):
            _, loss_data, w = sess.run([train_op, loss_op, W], feed_dict={X:X_data, y:y_data})
            loss_datas.append(float(loss_data))
            if e % 10== 0:
                loss, w = sess.run([loss_op, W],feed_dict={X:X_data, y:y_data})
                log_str = "Epoch %d \t Loss = %.4g \t Model:y = (%.4g)x1 + (%.4g)x2 + (%.4g)"
                print(log_str % (e, loss, w[1], w[2], w[0]))
    writer.close()
    # 可视化loss，其实也可以通过TensorBoard来展示Loss的变化情况
    print(loss_datas)
    import matplotlib.pyplot as plt
    import seaborn as sns
    #sns.set(context="notebook", style="whitegrid", palette="dark")
    #ax = sns.lmplot(x = "epoch", y = "loss", data=pd.DataFrame({"loss":loss_datas,"epoch":np.arange(epoch)}))
    #ax.set_xlabels("Epoch")
    #ax.set_ylabels("Loss")
    plt.figure()
    plt.grid(True)
    plt.plot(loss_datas)
    plt.xlabel("loss")
    plt.ylabel("epoch")
    plt.show()
LinearRegression_model()



