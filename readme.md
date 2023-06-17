##  接口调用说明

按照要求，我们实现了接口类AiGcMn，用于生成基于MNIST手写数字集的图像生成。

###  一、目录及文件

模型aigcmn.py会在当前目录下生成三个文件夹，分别是：

- **./data**：用于存放下载的MNIST数据集
- **./CGAN_images**：用于存放训练中生成的图像
- **./CGAN**：用于存放训练过程中实时保存的模型和最终训练完成的模型，该路径下的文件夹**CGAN/generator存放了调用接口函数进行实例化时，生成的输出图像**

###  二、可调用函数

- **AiGcMn.train()**：
  - 参数为self，无返回值；
  - 功能：训练模型并将模型保存在当前目录下。在第一次训练完成后可不再调用。
- **AiGcMn.generate()**：
  - 参数为self、input，返回值为output，其中input接受整数型n维tensor，每个整数在0~9之间，output返回为[n\*1\*28\*28]维tensor；
  - 功能：函数加载目录下训练好的模型，根据输入数字输出对应的手写字图像。

### 三、示例代码

以下为简单的示例代码，用于调用AiGcMn接口类并根据输入的数字生成图像：

```
# 实例化接口类
aigc = AiGcMn()

# 调用训练函数，训练模型
aigc.train()

# 训练完成后调用模型，输入input为n维tensor
# 第一次训练之后，可以直接加载并调用
# input测试为0~9是个
input = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
output = aigc.generate(input)
# 输出output为n*1*28*28维tensor
print(output.size())
```

运行aigcmn.py，将会执行此段代码进行实例化和输出测试。
