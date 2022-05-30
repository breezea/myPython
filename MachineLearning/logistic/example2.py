import numpy as np
import pandas
import matplotlib.pyplot as plt

def load(path):
    # 读取csv文件
    data = pandas.read_csv(path,header=None)
    return np.array(data.loc[:38,:159],dtype='float64'),np.array(data.loc[:38,160],dtype='float64')
# 标准化
def normalize(x):
    return (x-np.min(x,axis=0,))/(np.max(x,axis=0)-np.min(x,axis=0))

def linear(theta,x):
    # print('theta and x:',theta.shape,x.shape)
    # print('result:',(x@theta)[0],(x@theta)[-1])
    return x@theta

def sigmoid(x):
    return 1/(1+np.e**(-x))

def delat_w(H,Y,X):
    left = H-Y
    temp =  []
    # print((left[0]*X[0]).shape)
    for i in range(H.shape[0]):
        # print(i)
        temp.append(left[i]*X[i])
    return (np.array(temp).reshape(H.shape[0],X.shape[1]).mean(axis=0)).reshape(X.shape[1],1)
    # return (H-Y)*X*alpha

def cost(H,Y):
    return -np.sum(Y*np.log(H)+(1-Y)*np.log(1-H))/H.shape[0]

def loss(H,Y):
    return np.sum(np.square(H-Y))

def predict(theta,x):
    return sigmoid(linear(theta,x))

data, label = load('./scene_data.csv')
for index, item in enumerate(label):
    if item == 1:
        label[index] = 0
    else:
        label[index] = 1

data = normalize(data)
label = label.reshape(label.shape[0],1)
train_data, train_label = data[1:37,], label[1:37]
test_data, test_label = np.r_[data[0:1], data[37:38]], np.r_[label[0:1,], label[37:38]]

#初始化权重w
w = np.array(np.random.uniform(size=(160,1)))-0.5
epochs = 200 
alpha = 0.5 
loss_history = []

for epoch in range(epochs):
    # 计算梯度
    H = sigmoid(linear(w,train_data))
    # 更新w
    w = w - alpha*delat_w(H,train_label,train_data)
    Loss = loss(train_label,H)
    loss_history.append(Loss)
    print('第{}代, 损失为{}'.format(epoch,Loss))
print('训练完成')

print('------------------测试集------------------')
H = sigmoid(linear(w, test_data))
print('网络的预测值为:\n',H)
print('真实标签为:\n',test_label)
# 绘制历史损失
plt.plot(loss_history)
plt.show()