# -*- coding:utf-8 -*-
import numpy as np
from keras.datasets import mnist  #将会从网络下载mnist数据集
from keras.utils import np_utils
from keras.models import Sequential  #序列模型
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN #keras中三种RNN  SimpleRNN,LSTM,GRU
from keras.optimizers import Adam
from keras.utils import plot_model

# 数据长度，一行有28个像素
input_size=28
# 序列长度，一共有28行
time_steps=28
# 隐藏层cell个数
cell_size=50


#载入数据
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#查看格式
#（60000,28,28）
print('x_shape:',x_train.shape)
#（60000）
print('y_shape:',y_train.shape)



#格式是（60000,28,28）
#格式是样本数，time_steps（序列长度），input_size（每一个序列的数据长度）
#如果数据是（60000,784）需要转成（60000,28,28）
#除以255是做数据归一化处理
x_train=x_train/255.0 #转换数据格式
x_test=x_test/255.0 #转换数据格式
#label标签转换成 one  hot 形式
y_train=np_utils.to_categorical(y_train,num_classes=10) #分成10类
y_test=np_utils.to_categorical(y_test,num_classes=10) #分成10类

#定义序列模型
model=Sequential()

#循环神经网络
#一个隐藏层
model.add(SimpleRNN(
    units=cell_size,  #输出
    input_shape=(time_steps,input_size), #输入
))

#输出层
model.add(Dense(10,activation='softmax'))



#定义优化器
#学习速率为10的负4次方
adam=Adam(lr=1e-4)


#定义优化器，损失函数，训练效果中计算准确率
model.compile(
    optimizer=adam, #sgd优化器
    loss='categorical_crossentropy',  #损失用交叉熵，速度会更快
    metrics=['accuracy'],  #计算准确率
)

#训练
#六万张，每次训练64张，训练10个周期（六万张全部训练完算一个周期）
model.fit(x_train,y_train,batch_size=64,epochs=10)

#评估模型
loss,accuracy=model.evaluate(x_test,y_test)

print('\ntest loss',loss)
print('\ntest accuracy',accuracy)

model.save('mnist2.h5')
model.summary()
plot_model(model, "net2.svg", show_shapes=True)