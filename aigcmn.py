import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import pickle
import copy
from torchvision.utils import save_image
import torchvision.utils as vutils

import matplotlib.gridspec as gridspec
import os
import sys

# 接口类，实现了模型的结构定义、训练过程和接口函数generate
class AiGcMn:
    def __init__(self):
        self.count = 0
        # gepoch 生成器的训练伦茨
        self.epoch = 71
        self.gepoch = 1

        def loadMNIST(batch_size):  # 加载mnist数据集
            trans_img = transforms.Compose([transforms.ToTensor()])
            trainset = MNIST('./data', train=True, transform=trans_img, download=True)
            testset = MNIST('./data', train=False, transform=trans_img, download=True)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
            return trainset, testset, trainloader, testloader

        # 创建文件夹
        #CGAN保存模型文件 | GAN/generator保存generate函数生成的图像 | CGAN_images保存训练过程中生成的图像
        if not os.path.exists('./CGAN'):
            os.mkdir('./CGAN')
        if not os.path.exists('./CGAN/generator'):
            os.mkdir('./CGAN/generator')
        if not os.path.exists('./CGAN_images'):
            os.mkdir('./CGAN_images')

        # 损失函数
        self.criterion = nn.BCELoss()
        self.num_img = 100
        self.z_dimension = 110
        self.D = self.discriminator()
        self.G = self.generator(self.z_dimension, 3136)  # 1*56*56
        self.trainset, self.testset, self.trainloader, self.testloader = loadMNIST(self.num_img)  # data
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.0003)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=0.0003)

    # 模型的训练过程
    def train(self):
        for i in range(self.epoch):
            b = 1
            for (img, label) in self.trainloader:
                print("batch:%d pic:%d/60 pic_all:%d/4260" % ((i + 1), b, i * 60 + b))
                b = b + 1
                if b == 60:
                    break
                # 将一个0~9的数字转换为10维的onehot向量作为输入的一部分
                labels_onehot = np.zeros((self.num_img, 10))
                labels_onehot[np.arange(self.num_img), label.numpy()] = 1
                img = Variable(img)
                real_label = Variable(torch.from_numpy(labels_onehot).float())  # 真实label为1
                fake_label = Variable(torch.zeros(self.num_img, 10))  # 假的label为0

                # compute loss of real_img
                real_out = self.D(img)  # 真实图片送入判别器D输出0~1
                d_loss_real = self.criterion(real_out, real_label)
                real_scores = real_out  # 真实图片放入判别器输出越接近1越好

                # compute loss of fake_img
                z = Variable(torch.randn(self.num_img, self.z_dimension))  # 随机生成向量
                fake_img = self.G(z)
                fake_out = self.D(fake_img)  # 判别器判断假的图片
                d_loss_fake = self.criterion(fake_out, fake_label)
                fake_scores = fake_out  # 假的图片放入判别器输出越接近0越好

                # 损失函数优化过程
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.zero_grad()  # 判别器D的梯度归零
                d_loss.backward()  # 反向传播
                self.d_optimizer.step()  # 更新判别器D参数

                # 生成器G的训练
                for j in range(self.gepoch):
                    z = torch.randn(self.num_img, 100)  # 随机生成向量
                    z = np.concatenate((z.numpy(), labels_onehot), axis=1)
                    z = Variable(torch.from_numpy(z).float())
                    fake_img = self.G(z)  # 将向量放入生成网络G生成一张图片
                    output = self.D(fake_img)  # 经过判别器得到结果
                    g_loss = self.criterion(output, real_label)  # 得到假的图片与真实标签的loss
                    # bp and optimize
                    self.g_optimizer.zero_grad()  # 生成器G的梯度归零
                    g_loss.backward()  # 反向传播
                    self.g_optimizer.step()  # 更新生成器G参数
                    temp = real_label

            # 每十轮，保存一次训练的模型
            if (i % 10 == 0) and (i != 0):
                print(i)
                torch.save(self.G.state_dict(), r'./CGAN/Generator_cuda_%d.pkl' % i)
                torch.save(self.D.state_dict(), r'./CGAN/Discriminator_cuda_%d.pkl' % i)
                self.save_model(self.G, r'./CGAN/Generator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型
                self.save_model(self.D, r'./CGAN/Discriminator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型
                if i == 70:
                    self.save_model(self.G, r'./CGAN/Generator.pkl')
                    self.save_model(self.D, r'./CGAN/Discriminator.pkl')
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                i+1, self.epoch, d_loss.item(), g_loss.item(),
                real_scores.data.mean(), fake_scores.data.mean()))
            #temp = temp.to('cpu')
            #_, x = torch.max(temp, 1)
            #x = x.numpy()
            # print(x[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]])
            # showimg(fake_img, count)
            # plt.show()
            self.count += 1

    #用于在屏幕上打印生成的图片
    def show(self, images):
        num_img = 1
        images = images.detach().numpy()
        images = 255 * (0.5 * images + 0.5)
        images = images.astype(np.uint8)
        plt.figure(figsize=(4, 4))
        width = images.shape[2]
        gs = gridspec.GridSpec(1, num_img, wspace=0, hspace=0)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
            plt.axis('off')
            plt.tight_layout()
        plt.tight_layout()
        # plt.savefig(r'drive/深度学习/DCGAN/images/%d.png' % count, bbox_inches='tight')
        return width

    #将生成的图片集合在一张并打印出来
    def show_all(self, images_all):
        num_img = 1
        x = images_all[0]
        for i in range(1, len(images_all), 1):
            x = np.concatenate((x, images_all[i]), 0)
        #print(x.shape)
        x = 255 * (0.5 * x + 0.5)
        x = x.astype(np.uint8)
        plt.figure(figsize=(9, 10))
        width = x.shape[2]
        gs = gridspec.GridSpec(10, num_img, wspace=0, hspace=0)
        for i, img in enumerate(x):
            ax = plt.subplot(gs[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
            plt.axis('off')
            plt.tight_layout()

    def save_model(self, model, filename):  # 保存模型
        state = model.state_dict()
        x = state.copy()
        for key in x:
            x[key] = x[key].clone().cpu()
        torch.save(x, filename)

    def showimg(self, images):
        images = images.detach().numpy()
        images = images[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]]
        images = 255 * (0.5 * images + 0.5)
        images = images.astype(np.uint8)
        grid_length = int(np.ceil(np.sqrt(images.shape[0])))
        plt.figure(figsize=(4, 4))
        width = images.shape[2]
        gs = gridspec.GridSpec(grid_length, grid_length, wspace=0, hspace=0)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
            plt.axis('off')
            plt.tight_layout()
        #  plt.tight_layout()
        # 创建文件夹
        if not os.path.exists('./CGAN_images'):
            os.mkdir('./CGAN_images')
        plt.savefig(r'./CGAN_images/%d.png' % self.count, bbox_inches='tight')



    #接口函数，输入为n维tensor，输出为n*1*28*28维tensor
    def generate(self, value):
        num_img = 1
        z_dimension = 110
        # 加载保存的模型
        D = self.discriminator()
        G = self.generator(z_dimension, 3136)
        D.load_state_dict(torch.load(r'./CGAN/Discriminator.pkl'))
        G.load_state_dict(torch.load(r'./CGAN/Generator.pkl'))
        lis = []
        input = value.numpy()
        input = input.astype("int32")
        isfirst = True
        for i in input:
            z = torch.randn((num_img, 100))  # 随机生成向量
            x = np.zeros((num_img, 10))
            x[:, i] = 1
            z = np.concatenate((z.numpy(), x), 1)
            z = torch.from_numpy(z).float()
            fake_img = G(z)  # 将向量放入生成网络G生成一张图片
            lis.append(fake_img.detach().numpy())
            # print(fake_img.size())
            if isfirst:
                images = fake_img
                isfirst = False
            else:
                images = torch.cat((images, fake_img), 0)  # 将生成的单一图片拼接到images中
            output = D(fake_img)  # 经过判别器得到结果
            AiGcMn.show(self, fake_img)
            plt.savefig('./CGAN/generator/%d.png' % i, bbox_inches='tight')

        self.show_all(lis)
        plt.savefig('./CGAN/generator/all.png', bbox_inches='tight')
        plt.show()

        return images

    def __call__(self):
        pass

    # 鉴别器子类
    class discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.dis = nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=1, padding=2),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d((2, 2)),

                nn.Conv2d(32, 64, 5, stride=1, padding=2),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d((2, 2))
            )
            self.fc = nn.Sequential(
                nn.Linear(7 * 7 * 64, 1024),
                nn.LeakyReLU(0.2, True),
                nn.Linear(1024, 10),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.dis(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # 生成器子类
    class generator(nn.Module):
        def __init__(self, input_size, num_feature):
            super().__init__()
            self.fc = nn.Linear(input_size, num_feature)  # 1*56*56
            self.br = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.gen = nn.Sequential(
                nn.Conv2d(1, 25, 3, stride=1, padding=1),
                nn.BatchNorm2d(25),
                nn.ReLU(True),

                nn.Conv2d(25, 1, 2, stride=2),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(x.size(0), 1, 56, 56)
            x = self.br(x)
            x = self.gen(x)
            return x


# 实例化接口类
aigc = AiGcMn()
# 调用训练函数，训练模型
aigc.train()
# 训练完成后调用模型，输入input为n维tensor
# 第一次训练之后，可以直接加载并调用
input = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
output = aigc.generate(input)
print("size of output:")
print(output.size())