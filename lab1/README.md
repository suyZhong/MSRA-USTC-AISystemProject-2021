## 环境	

|||Mac (local)|Bitahub|
|--------|--------------|--------------------------|--------------------------|
|硬件环境|CPU（vCPU数目）|6|2|
||GPU(型号，数目)|无|gtx1080Ti|
|软件环境|OS版本|MacOS Catalina|Maybe Ubuntu|
||深度学习框架<br>python包名称及版本|Pytorch1.5 & Python3.7.4|Pytorch1.5 & Python|
||CUDA版本|无||
|||||

## 实验流程

大体流程为，先在mac上测试程序没有bug，然后写shell脚本将所有流程整合并上传到bitahub平台进行运行。

1. 运行MNIST样例程序

2. 修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。

   1. 首先添加代码段

      ```python
      writer = SummaryWriter(args.output)
      ...
      if args.save_graph:
              writer.add_graph(model, data)
      ```

   2. 使用如下指令开始运行

      ```shell
      python mnist_stu.py --dataset DATASET --save-graph --output OUTPUT
      ```

3. 记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

   1. 添加代码段

      ```python
      # 在train、函数里
      if batch_idx % args.log_interval == 0:
          # 已经进行tb画图了，所以这里注释掉
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          #     epoch, batch_idx * len(data), len(train_loader.dataset),
          #     100. * batch_idx / len(train_loader), loss.item()))
          if args.save_scalar and batch_idx != 0:
              writer.add_scalar('training loss', running_loss / args.batch_size * args.log_interval, epoch * len(train_loader)+ batch_idx)
              writer.add_scalar('training_accuracy', correct / args.batch_size * args.log_interval, epoch * len(train_loader) + batch_idx )
          ...
      ```

   2. 使用如下指令开始运行

      ```shell
      python mnist_stu.py --dataset DATASET --save-scalar --output OUTPUT
      ```

4. 添加神经网络分析功能

   1. 添加代码段

      ```python
      if args.profile:
          with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
              model(data[0].reshape(1, 1, 28, 28))
          print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
      ```

   2. 使用如下指令运行

      ```shell
      python mnist_stu.py --profile --dataset DATASET --output OUTPUT
      ```

   3. 更改batch_size进行运行

      ```shell
      python mnist_stu.py --profile --dataset DATASET --output OUTPUT -batch_size 16
      ```

5. 撰写[脚本](src/run_all.sh)，编写[dockerfile](../resources/dockerfile)，并上传到bitahub进行运行和测试

## 实验结果

这里展示上传bitahub后的一键脚本跑出的结果。

#### 1.模型可视化

##### 神经网络数据流图

<img src="images/lab1-graph.png" alt="lab1-graph" style="zoom:50%;" />

##### 损失和正确率趋势图

<img src="images/train.png" alt="train" style="zoom:33%;" /><img src="images/loss.png" alt="train" style="zoom:33%;" />

##### 网络分析（bs=64）

使用bitahub进行profile分析会出现一个bug。即在训练前进行测试，卷积层上会花费巨额的时间，如下图：

use_cuda profile

![profile1](images/profile1.png)

use_cpu profile

![profile_cpu_bita](images/profile_cpu_bita.png)

都不同程度的在convolution花费时间很大，这很不寻常。

而在mac上使用cpu进行profile如图

![image-20210525134836739](images/image-20210525134836739.png)

或者，在训练后再对模型进行profile：

![image-20210525134959559](images/image-20210525134959559.png)

则看起来符合预期。出现这样结果的原因推测还是bitahub的问题。

#### 2.网络分析

