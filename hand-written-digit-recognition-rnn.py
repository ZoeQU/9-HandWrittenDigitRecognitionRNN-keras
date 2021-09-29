# -*- coding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils import plot_model

TIME_STEPS = 28  # 数据长度，一行有28个像素
INPUT_SIZE = 28  # 序列长度，一共有28行
BATCH_SIZE = 50  # 隐藏层cell个数
index_start = 0
OUTPUT_SIZE = 10  # 分成10类 [0,1,2,3,4,5,6,7,8,9]
CELL_SIZE = 75
LR = 1e-3

# 载入数据
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 查看格式
print('X_shape:',X_train.shape)  #（60000,28,28）
print('Y_shape:',Y_train.shape)  #（60000）

X_train = X_train.reshape(-1, 28, 28)/255  # 数据归一化处理,转换数据格式
X_test = X_test.reshape(-1, 28, 28)/255  # 数据归一化处理,转换数据格式

Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
# 定义序列模型
model = Sequential()

# 一个隐藏层
model.add(SimpleRNN(batch_input_shape=(BATCH_SIZE,
                                       TIME_STEPS,
                                       INPUT_SIZE),
                    output_dim=CELL_SIZE))
# 输出层
model.add(Dense(OUTPUT_SIZE, activation='softmax'))
# 学习速率为10的负3次方
adam = Adam(LR)

# 定义优化器，损失函数，训练效果中计算准确率
model.compile(
    loss='categorical_crossentropy', #损失用交叉熵，速度会更快
              optimizer=adam, #sgd优化器
              metrics=['accuracy'] #计算准确率
)

# train
# for i in range(500):
#     X_batch = X_train[index_start:index_start + BATCH_SIZE,:,:]
#     Y_batch = Y_train[index_start:index_start + BATCH_SIZE,:]
#     index_start += BATCH_SIZE
#     cost = model.train_on_batch(X_batch,Y_batch)
#     if index_start >= X_train.shape[0]:
#         index_start = 0
#     if i % 100 == 0:
#         # 评估模型
#         cost, accuracy = model.evaluate(X_test, Y_test, batch_size=50)
#         # W,b = model.layers[0].get_weights()
#         print('accuracy:', accuracy)

# 等同于上述代码，但要保证train和test的batch_size相同
model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=5)
cost, accuracy = model.evaluate(X_test, Y_test,batch_size=BATCH_SIZE)
print('accuracy:', accuracy)

model.save('mnist.h5')
model.summary()
plot_model(model, "net.svg", show_shapes=True)