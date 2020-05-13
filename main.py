import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import models
import config

import json
from PIL import Image

#数据增强
train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=156, scale=(0.8, 1.0)), #随机裁剪到156*156
        transforms.RandomRotation(degrees=15), #随机旋转
        transforms.RandomHorizontalFlip(), #随机水平翻转
        transforms.CenterCrop(size=124), #中心裁剪到124*124
        transforms.ToTensor(), #转化成张量
        transforms.Normalize([0.485, 0.456, 0.406], #归一化
                             [0.229, 0.224, 0.225])
])

test_valid_transforms = transforms.Compose(
        [transforms.Resize(156),
         transforms.CenterCrop(124),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
])

#利用Dataloader加载数据
train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES

train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

'''
#获取索引到类名的映射，以便查看测试影像的输出类
idx_to_class = {v: k for k, v in train_datasets.class_to_idx.items()}
print(idx_to_class)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)
'''

valid_datasets = datasets.ImageFolder(valid_directory,transform=test_valid_transforms)
valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

#print(train_data_size, valid_data_size)

#使用Resnet-50的预训练模型进行迁移学习
resnet50 = models.resnet50(pretrained=True)

#查看更改前的模型参数
#print('before:{%s}\n'%resnet50)

for param in resnet50.parameters():
    param.requires_grad = False #冻结预训练网络中的参数

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256), #将resnet50最后的全连接层输入给256输出单元的线性层
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 11),
    nn.LogSoftmax(dim=1) #输出11通道softmax层
)
#查看更改后的模型参数
#print('after:{%s}\n'%resnet50)

#定义损失函数和优化器
loss_func = nn.NLLLoss()
optimizer = optim.SGD(resnet50.parameters(), lr = 0.01)

#训练过程
def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs): #训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train() #训练

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)

            optimizer.zero_grad() #梯度清零

            outputs = model(inputs) #数据前馈，正向传播

            loss = loss_function(outputs, labels) #输出误差

            loss.backward() #反向传播

            optimizer.step() #优化器更新参数

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval() #验证

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  : #记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, 'trained_models/resnet50_model_' + str(epoch + 1) + '.pth')
    return model, record

'''
def predict(model, test_image_name):

    transform = test_valid_transforms

    test_image = Image.open(test_image_name).convert('RGB')
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 124, 124).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 124, 124)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        start = time.time()
        out = model(test_image_tensor)
        stop = time.time()
        print('cost time', stop - start)
    ps = torch.exp(out)
    topk, topclass = ps.topk(3, dim=1)
    names = []
    for i in range(3):
        names.append(cat_to_name[idx_to_class[topclass.cpu().numpy()[0][i]]])
        print("Predcition", i + 1, ":", names[i], ", Score: ",
              topk.cpu().numpy()[0][i])
        plt.barh([2, 1, 0], topk.cpu().numpy(), tick_label=names)
'''

#结果
if __name__=='__main__':
    num_epochs = config.NUM_EPOCHS
    trained_model, record = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
    torch.save(record, config.TRAINED_MODEL)

    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')
    plt.show()

'''
    model = torch.load('trained_models/resnet50_model_23.pth')
    predict(model, '61.png')
'''