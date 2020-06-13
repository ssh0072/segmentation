from tqdm import tqdm  # 进度条工具
import os
import shutil
import torch.nn.functional as F  # 调用softmax
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import time
from imgaug import augmenters as iaa
# 导入自定义的库
# from utils.image_process import LaneDataset, ImageAug, DeformAug
# from utils.image_process import ScaleAug, CutOut, ToTensor
# from utils.loss import MySoftmaxCrossEntropyLoss
# from model.deeplabv3plus import DeeplabV3Plus
# from model.unet import ResNetUNet
# from config import Config
# from utils.metric import compute_iou


# train_net_name = 'deeplabv3p'
# train_net_name = 'unet'
# nets = {'deeplabv3p': 1, 'unet': ResNetUNet}

class Config(object):
    # 模型参数配置  ########################################
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8  # 类别数
    IMG_SIZE = (768, 256)  # 分阶段训练的分辨率；分别为(768，256),(1024,384),(1536,512);最后一个阶段训练完后直接resize到原图分辨率大小！！！！！！！！
    # 训练参数(超参数)配置  #################################
    EPOCHS = 1  # 训练轮数
    train_batch_size = 4  # 训练集每批大小
    val_batch_size = 1  # 验证集集每批大小
    WEIGHT_DECAY = 1.0e-4  # 权重衰减参数
    BASE_LR = 0.0006  # 基础学习率
    LR_MIN = 1e-6  # 最小学习率
    # 存储路径  ######################################
    save_log_path = "logs"  # 日志（训练和验证结果）保存路径
    save_parameter_path = "model_weight"
    # 训练设备环境
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cup")

def main():
    prev_time = time.time()
    # 设置model parameters
    config = Config()
    # 设置训练网络
    train_net = 'unet'
    # 是否使用预训练的模型参数
    use_trained = True

    # 检查保存日志的路径是否存在
    if not os.path.exists(config.save_log_path):
        # 删掉
        # shutil.rmtree(config.SAVE_PATH)
        # 建立一个新的单级文件夹～(注意不是文件)
        os.makedirs(config.save_log_path, exist_ok=True)  # 在当前路径下创建文件夹,名为logs

    # 检查保存模型参数的路径是否存在
    if not os.path.exists(config.save_parameter_path):
        # 删掉
        # shutil.rmtree(config.SAVE_PATH)
        # 建立新的多级文件夹～
        os.makedirs(os.path.join(config.save_parameter_path,train_net), exist_ok=True)  # 在当前路径下创建多级文件夹
    
    """
    文件操作流程：
    open、write、flush、close
    open函数：若文件存在，会清空其原有内容（覆盖文件）；反之，则创建新文件～
    """
    # 打开记录训练结果的csv文件，如果不存在则创建
    log_train = open(os.path.join(config.save_log_path, "train.csv"), 'a')  # a为追加
    # log_train = open(os.path.join(config.SAVE_PATH, "train.csv"), 'w')  # 创建可写状态的train.csv，路径为log/train.csv
    # 打开记录验证结果的csv文件，如果不存在则创建
    log_test = open(os.path.join(config.save_log_path, "test.csv"), 'a')  # a为追加
    # log_test = open(os.path.join(config.SAVE_PATH, "test.csv"), 'w')  # 返回一个可写状态的File对象；w为清空后重写

    # 1，获取训练数据集：train dataset
    # 调用自定义的LaneDataset类，建立带索引的图片和label对应，同时做数据预处理和数据增强操作
    #train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5),ToTensor()]))
    train_list = [RandomHFlip(), RandomVFlip(), ImageAug(), CutOut(mask_size=100, p=0.5), ToTensor()]
    train_dataset = LaneDataset("train.csv", config, transform=train_list)
    # dataframe, size为读取的csv数据数(整个数据集size)
    """
    返回的是一个LaneDataset对象;读取的是一个2维dataframe的csv文件；默认就自动为每个维度里的路径值加上索引
    然后对读取的图片做预处理操作，最终形成二维dict
    其中key分别为'image'，'mask'，对应的value分别是tensor.float32类型,tensor.int64类型
    """
    # 2，生成可批量读取数据集的迭代器：train dataset‘s dataloader：做洗牌操作
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    # 创建DataLoader对象train_data_batch，批大小设置为BATCH_SIZE
    train_data_batch = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, **kwargs)
    """
    返回DataLoader对象，即数据迭代器，其个数（长度）为全部数据集大小/batch_size
    DataLoader按batch取数据底层原理： _next_data
    通过_next_index()函数获取下一个索引，再通过_dataset_fetcher.fetch(index)函数根据索引获取LaneDataset中的图片路径；
    当数据量够一个批的大小时，就停止不取了
    """
    # 1，获取验证数据集：validation dataset
    valid_list = [ToTensor()]
    val_dataset = LaneDataset("val.csv", config, transform=valid_list)
    # 2，批量读取数据集：validation dataset‘s dataloader：不做洗牌操作;验证集bath_size默认取1
    val_data_batch = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, drop_last=False, **kwargs)

    # 3，建立网络实例：作为参数传入到train函数和test函数中
    my_net = ResNetUNet(config)  # config为网络模型的参数
    if torch.cuda.is_available():
        my_net = my_net.cuda(device=config.device)  # 将模型放在GPU上
        # 在这里加了一个数据并行，相当于甲类一个moduel
        my_net = torch.nn.DataParallel(my_net, device_ids=config.device)
    if use_trained:  # 加载模型参数；可以在之前小分辨率训练的结果上输入大分辨率继续训练！！！！！！！！！！！！！！！
        my_net.eval()
        model_path = os.path.join(os.getcwd(), 'model_weight', 'unet', 'finalNet.pth.tar')
        my_net.load_state_dict(torch.load(model_path)['state_dict'])

    # 4，定义优化器：Adam、SGD、...
    optimizer = torch.optim.Adam(my_net.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)

    # 5，循环调用train函数训练、并调用test函数验证：
    for epoch in range(config.EPOCHS):
        # 预热学习率
        # adjust_lr(optimizer, epoch)
        # 调用train函数训练
        train(my_net, epoch, train_data_batch, optimizer, log_train, config)  # 里面循环多次，分批取整个数据集中的数据，最后算平均IOU
        # 调用test函数进行验证
        valid(my_net, epoch, val_data_batch, log_test, config)
        # 模型参数保存torch.save
        # print(my_net.state_dict())  #返回的是一个OrderDict，存储了网络结构的名字和对应的参数
        if epoch+1 % 2 == 0:  # 每几轮存储一次模型参数
            # 每两轮，就只存储net模型参数，不存储网络结构（所以加载时要先建好网络再加载）
            torch.save({'state_dict': my_net.state_dict()},
                       os.path.join(os.getcwd(), 'model_weight', train_net, "laneNet{}.pth.tar".format(epoch+1)))
    # ？为什么不是 torch.save(net.state_dict(), os.path.join(os.getcwd(), config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
    # 关闭文件
    log_train.close()
    log_test.close()
    # 训练完后存储最终的模型参数
    #torch.save({'state_dict': my_net.state_dict()}, os.path.join(os.getcwd(), 'model_weight', train_net, "finalNet.pth.tar"))
    # 计时
    end_time = time.time()
    print('time cost:{}'.format(end_time-prev_time))

# 定义train函数：循环batch，进来一个batch的数据，计算一次梯度，更新一次网络
def train(net, epoch, train_data_batch, optimizer, log_train, config):  # train_data_batch是按batch_size打包好的数据迭代器
    net.train()  # 转换为训练状态
    total_train_loss = 0.0
    # 创建混淆矩阵mc：tensor.float32，8 x 8
    confusion_matrix = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES)).to(config.device)  # tensor.float32, 8x8; 如果知道GPU设备，则需要再加上.to(device)
    # 迭代器加入进度条：i/t，其中当前第i个batch，全部数据集被分成的t个batch；当全部数据集中数据按batch取完时，进度条就显示满格
    dataprocess = tqdm(train_data_batch)
    # 利用迭代器批量读取数据： 每批计算一次loss（一批可以是一个数据，也可以是若干个数据）,直至数据集中的数据被取完为止！！
    # 真正开始读数据，以及做数据处理工作：分批读入图片并"喂进"模型训练的目的，占用内存小
    for batch_item in dataprocess:  # 一共有整个数据集大小/batch_size个批，每批中打包了batch_size条数据！！！！！！！
        """
        dataprocess：加了进度条的迭代器对象，里面均为按批大小batch_size分割的一个个的批数据集（image和label成对组成的dict）
        因此，进去的数据CHW和HW，读出来后会自动打包为BCHW和BHW，这里不需要自己手动增加维度！！！！！！！！！！！！！！！！！！
        dataprocess的个数为整个数据集大小/batch_size
        batch_item：二维dict
        key分别为'image','mask'；
        value分别为tensor.float32,BCHW; (2,3,256,768)    tensor.int64,BHW; (2,256,768)   当batch_size=2时
        源于LaneDataset函数对数据的处理，之所以要这样是数据类型是因为因为后面要输进的函数SoftmaxCrossEntropyLoss需要！！！！
        """
        # 取出每批中每个图片的路径（key对应的value值）
        image, label = batch_item['image'], batch_item['mask']  # tensor.float32,NCHW; tensor.int64,NHW; (2,3,256,768), (2,256,768);因为batch_size=2，所以一取取两个！！！！！！！！！
        if torch.cuda.is_available():
            image = image.cuda()
            # image = image.to(config.device)  # 法2：其中device=torch.device("cuda:0" if torch.cuda.is_available() else "cup")
            label = label.cuda()
        optimizer.zero_grad()   # 每一批计算一次求导，更新一次参数；计算导数时，先清零，否则会在之前的基础上累加，相当于扩大batch_size;
        """
        PyTorch默认会对梯度进行累加，如果不手动设置梯度清零，梯度累加相当于变相扩大batch！！！！！！！！！！！
        如果默认梯度累加，学习率也要适当放大
        将每个parameter的梯度清0，原因：
        在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，因此需要对每个batch调用一遍zero_grad（）将参数梯度置0
        """
        # 送进网络，输出预测分数
        out = net(image)  # tensor.float32,NCHW (2,8,256,768)；过程：image(1,3,1701,3384)——>out(1,8,1701,3384),8为类别个数；传入forward函数里的参数，默认调用forward函数;
        # 计算交叉熵loss作为优化的目标函数
        train_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, label)  # 返回tensor.float32，标量
        #  train_loss = SemanticSegLoss('cross_entropy+dice')(out, label.type(torch.long))
        """
        由于MySoftmaxCrossEntropyLoss是一个class，因此先创建该类的对象，即类名（init中的参数）
        然后再由该对象调用该类中的forward函数，传入forward中的参数
        """
        total_train_loss += train_loss.item()  # train_loss是tensor，所以需要用item()取出里面的值！！！！！！！！
        # total_loss += loss.detach().item()  # 加detach的目的：梯度截断，减少内存？
        # 向后计算
        train_loss.backward()
        # 更新参数
        optimizer.step()
        # 计算模型评价指标miou：
        # 步骤：先计算出混淆矩阵，最后再根据不断累加的混淆矩阵计算miou
        # 方法1：首先要得出网络输出的最大的预测概率值对应的位置（即train_id 类别），目的：构建混淆矩阵
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)  # tensor.int64, NHW；过程：tensor.float32(2,8,384,1024)——>tensor.int64(2,384,1024)
        # label = label.squeeze(1)
        """
        注意：label是否做处理，分两种情况，因为送入到label的维度要和pred维度一致
        当label为3通道彩色图时，label也要减少一个C维度，变成NHW
        label = label.squeeze(1)  
        如果label为灰度图，本来就是NHW，不需要再减少
        """
        # 然后构建混淆矩阵，并累加：此时输进去的pred和label均为tensor.int64,NHW形状（因为pred经过了softmax和argmax）
        confusion_matrix += get_confusion_matrix(pred, label)  # tensor.float32, num_classes x num_classes
        # 方法2：计算出每个类别i（每种类别都看作二分类，正类和负类）的TP等值
        # result = compute_iou(pred, mask, result)
        # 设置进度条上显示的信息
        dataprocess.set_description_str("epoch:{}".format(epoch+1))
        dataprocess.set_postfix_str("train_loss:{:.4f}".format(train_loss.item()))
    # 当整个数据集都走完后：
    # 计算整个数据集的平均损失值：即所有批得出的损失值累加后除以批的个数
    mean_train_loss = total_train_loss / len(train_data_batch)  # train_data_batch为加载器对象, len(train_data_batch)=1; 即整个数据及数量/batch_size=2/2=2
    # 注意：len(train_data_batch)为加载器对象的个数，即批的个数，即全部数据集大小/batch_size
    # 计算整个数据集所有类别预测值(混淆矩阵)的IOU的平均值：整个数据集跑下来，各个类别的IOU加和后除以类别的个数
    mean_iou = get_miou(confusion_matrix)  # tensor.float32，一维标量
    """对应上面方法2：
    iou = np.zeros(8)
    for i in range(8):
        iou[i] = result["TP"][i]/result["TA"][i])
    miou = np.nanmean(iou)   """
    # 打开指定的文件，并记录数据迭代了多少次
    log_train.write("Epoch:{}, train loss is {:.4f}, mean_iou is {:,.4f}\n".format(epoch, mean_train_loss, mean_iou))
    # 刷新到硬盘，同时情况缓冲区
    log_train.flush()  # flush()方法用来把文件从内存buffer（缓冲区）中强制刷新到硬盘中～，同时清空缓冲区

# 优化学习率：技巧1，warming up，预热学习率
def adjust_lr(optimizer, epoch, warm_epochs):
    """
    训练trick：预热阶段warming up + 训练阶段learning rate decay
    背景：
    在训练开始的时候先选择使用一个较小的学习率（预热阶段），训练了一些epoches（预热周期）后，再修改为预先设置的学习来进行训练（训练阶段）
    但，预热学习率完成后的训练过程，学习率是逐步衰减的，有助于使模型收敛速度变快，效果更佳
    因此，学习率的整体趋势是先升高再下降
    原因：
    1>预热阶段设置小的学习率原因：
    刚开始训练时,模型的权重(weights)是随机初始化的，如果选择一个较大的学习率,可能带来模型不稳定(发生振荡)，
    选择Warmup预热学习率的方式，在小学习率下，模型可以慢慢趋于稳定,等模型相对稳定后再选择预先设置的学习率进行训练
    2>训练过程训练率大，且逐步衰减的原因：
    大的学习率可以加速收敛，适合探索更旷阔的空间；后期采用小的学习率有助于接近最优解

    法1：constant warm up
    即：每隔若干个周期，学习率衰减一次，即原来学习率*衰减率（如取0.1）
    缺点：如果从一个很小的学习率一下变为比较大的学习率可能会导致训练误差突然增大，而引起的训练不稳定需要Optimizer用几个迭代去稳定它

    法2：gradual warm up
    前3个epochs采用默认参数训练（lr=0.001），
    在之后3个epochs的训练中，每个epoch平均分配出6个改变lr的地方？？，改变方式为:0.001-0.0006-0.0003-0.0001-0.0004-0.0008-0.001
    最后两个epoch一般采用0.0004-0.0001之间学习率训练的策略
    由于测试集与训练集在图像质量和视觉感知上差距不小，太小的学习率很容易导致过拟合，所以最小的学习率采用0.0001
    在8-10个epochs后，训练基本上就结束了
    # def _adjust_lr(self, epoch, iter_no, iter_count):
    if epoch <= warm_epochs:  # 前几个epoch逐渐升高学习率
        rate = ((epoch - 1) * iter_count + iter_no) / (warm_epochs * iter_count)
        lr = Config.LR_MIN + (Config.LR - Config.LR_MIN) * rate
    """
    # constant warm up
    if epoch == 0:
        lr = 1e-4  # 小
    elif epoch == 10:  # 10为预热周期
        lr = 1e-2  # 大；*10
    elif epoch == 30:
        lr = 1e-3  # 小；*0.1
    elif epoch == 50:
        lr = 1e-4  # 小; *0.1
    else:
        return
    # optimizer设置lr参数，否则更新不到！！！！！！！！！！！
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 赋值到网络学习率这个参数变量上

def valid(net, epoch, val_data_batch, log_test, config):  # 不需要调用优化器，只需要计算损失函数和评价指标值（需要输出最终预测的类）
    net.eval()   # 将model转换成eval,训练状态
    total_test_loss = 0.0
    # 创建混淆矩阵mc,这里是8x8大小
    confusion_matrix = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES)).to(config.device)  # 如果知道GPU设备，则需要再加上.to(device)
    dataprocess = tqdm(val_data_batch)  # val_data_batch迭代器对象个数为2（=整个数据集大小/batch_size）
    # result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
    # 返回嵌套的dict，{'TP':{0:0, 1:0, 2:0, .. ,7:0} , 'TA':{0:0, 1:0, 2:0, .. ,7:0}}
    with torch.no_grad():  # 测试阶段，不需要计算梯度，节省内存；不加这个就会报错！！！！！！！！！！！！！！！！！！！！！！！！！！
        for batch_item in dataprocess:
            image, label = batch_item['image'], batch_item['mask']  # 因为batch_size=1,所以(1,3,256,768)tensor.float32; (1,3,256,768)tensor.int64
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            # 模型输出各类的预测分数；计算图通常会在使用网络时生成，在测试阶段没有张量需要做梯度更新，不需要保存任何图，因此需要自己手动设置不需要梯度更新！！！！！！！！！
            out = net(image)  # out形状(1,8,1710,3384)
            # 损失函数
            test_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(out, label)
            # 损失函数累加到total_test_loss变量中，detach()作用：截断反向传播的梯度流,查detach用法？？？？？？
            total_test_loss += test_loss.detach().item()   # 要加detach()，否则报错！！！！！！！！！！！！！
            # 最终预测的类，即预测结果中概率最大的类
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)   # pred形状(1,1710,3384)
            # 计算模型评价指标iou：根据预测值perd和label（放在CPU上计算）
            # iou = compute_iou(pred, label, result)  # 为什么不是求平均iou????
            confusion_matrix += get_confusion_matrix(pred, label)
            # 设置进度条
            dataprocess.set_description_str("epoch:{}".format(epoch+1))
            dataprocess.set_postfix_str("test_loss:{:.4f}".format(test_loss))  # 进度条上显示每批计算得出的损失值
    # 计算整个数据集的平均损失值：即所有批得出的损失值累加后除以批的个数
    mean_test_loss = total_test_loss / len(val_data_batch)  # len(val_data_batch)=2; 即整个数据及数量/batch_size=2/1=2
    # 计算整个数据集所有类别预测值的IOU的平均值：整个数据集跑下来，各个类别的IOU加和后除以类别的个数
    mean_iou = get_miou(confusion_matrix)
    """
    # 记录每一个类别的iou，目的：求平均iou
    for i in range(8):
        result_string = "{}: {:.4f} \n".format(i, iou[i])
        print(result_string)
        # 将写入log文件
        testF.write(result_string)
    """
    # 将当前训练的轮数，平均损失值写入到log文件
    log_test.write("Epoch:{}, test loss is {:.4f} ,mean_iou is {:,.4f}\n".format(epoch, mean_test_loss, mean_iou))
    log_test.flush()  # 当需要立刻写入到文件时使用；用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
######################################################################################

# 创建Dataset子类～
class LaneDataset(Dataset):  # 以dataset为父类，声明一个子类aneDataset车道数据集

    # __init__ 将所有的数据都加载进来，传入csv文件和图像增强方法transform
    def __init__(self, csv_file, config, transform=None):
        super(LaneDataset, self).__init__()
        # 1,读取csv文件（两列，第一列为图片路径，第二列为label路径），获取图片和label路径list
        # header必须等于0，否则标注的索引，0索引读取的数据为列名，并不是真实数据的路径值！！！！！！！！！！！
        self.data = pd.read_csv(os.path.join('/Users/ssh/Downloads/d2l-zh/data_list', csv_file), header=0,
                                names=["image", "label"], nrows=4)  # dataframe类型，形状(nrows,2)；os.getcwd()返回当前工作目录，指所运行脚本的目录～
        print('{} Data Size:{}'.format(csv_file, len(self.data)))
        """
        pd.read_csv函数参数解释：
        第一个参数为文件路径
        header=None，表示自动生成0，1，2，...为列名，文件从第0行就读取为数据
        names=[a,b,c,...]，表示给表加列名,作为列索引～
        """
        # 将图像的地址加载进来，需要保证一条真实的数据对应一个真实的index
        # 这两列对应的是真实值的地址，用于后面__getitem__函数给路径建立索引～～
        self.images = self.data["image"].values  # ndarray, nrows维行向量；返回读取的图片路径～
        # 如果上面pd.read_csv中header=None，则这里索引从1开始取！！！！！！！！！！！！！！！！！！
        # self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values  # ndarray, nrows维行向量；返回读取的label路径～
        # 2,图像增强方法
        self.transform = transform
        self.config = config
    # 数据集的大小：在数据生成过程中，可以帮助设置batch_size的大小，和epoch
    def __len__(self):
        return self.labels.shape[0]  # 总长度
    # 索引——>（图片和label）地址——>(图片和label)图片数据——>预裁剪、标签encoder转换为训练标签、数据增强
    def __getitem__(self, idx):
        # 1,读取图像和标签图像，其中idx是数据生成过程系统自动建立的索引
        ori_image = cv2.imread(self.images[idx])  # ndarray.uint8, HWC ; cv2.imread根据路径读取相应图片，返回ndarray.uint8
        ori_label = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)  # ndarray.uint8, HW; 如果不加cv2.IMREAD_GRAYSCALE，默认读取的是3通道，返回HWC
        """
        读取图片3种方法：
        cv2.imread(path)，返回ndarray类型
        matplotlib.image.imread(path)，返回ndarray类型
        PIL.image.open(pqth),返回PIL image对象
        注意：如果后续对里面的数据进行操作，需要转换成ndarray类型，操作完后再转换成PIL image对象类型
        如下：
        ori_label = PIL.image.open(self.labels[idx]）
        ori_label = np.asarray(ori_label)  # 从PIL Image类型转换为ndarray类型，为了做下一步转换成train_id操作
        train_label = id_to_trainid(ori_label)  # label的Id转换为TrainId，这一步必须加上
        train_label = PIL.Image.fromarray(train_label.astype(np.uint8))  #从ndarray类型转换为PIL Image类型
        """
        # 2,预裁剪图像和标签图像：目的是去掉无用信息，不是数据增强～～
        train_img, train_label = crop_resize_data(ori_image, ori_label, self.config.IMG_SIZE)  # ndarray.uint8,HWC; ndarray.uint8,HW
        # 3,Encode转换为训练使用的label（在数据增强前！！！！！！！！！！！）
        train_label = encode_labels(train_label)  # nddary.float64, HW; 过程：ndarray.uint8——>ndarray.float64
        sample = [train_img.copy(), train_label.copy()]  # 二维list数组: image为 nddary.uint8,HWC; label为 ndarray.float64,HW
        # 4,数据增强操作：输入要网球前要转化为tensor，否则输入到网络中要报错！！！
        if self.transform:
            for t in self.transform:
                sample = t(sample)
        return sample  # dict二维字典：image为 tensor.float32,CHW; label为 tensor.int64,HW; 过程：二维list——>二维dict;ndarray——>tensor

# crop the image to discard useless parts：裁剪掉没用的部分; image和label始终ndarray.uint8
# 经观察图片上方h方向第690个像素点往上均没有正样本存在，offset设为690，即原图可以默认为(690:,:)，同时label也要对应改变～
# 后续在此基础上再做裁剪，用于分阶段训练，送入到网络中做数据增强、训练～～
def crop_resize_data(image, label=None, image_size=None, offset=690):  # image_size=(768, 256)为WH！！！！！！！！！
    roi_image = image[offset:, :]  # nddary.uint8,HWC; 过程：(1710,3384,3)——>(1020,3384,3)
    # 判断要裁剪的区域对应的label是否存在：如果存在，label也要做相同的裁剪；否则只需要裁剪image～
    if label is not None:
        # 裁剪的label目标区域是原label的一部分
        roi_label = label[offset:, :]  # nddary.uint8,HW; 过程：(1710,3384)——>(1020,3384)
        # 调用cv2库中的resize函数裁剪为(1024, 384)大小
        # 对image用线性插值：roi_image为目标区域，image_size裁剪后的大小
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)  # ndarray.uint8, HWC；过程：(1020,3384,3)——>(256,768,3)
        # 对label用最邻近插值
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)  # ndarray.uint8, HW；过程：(1020,3384)——>(256,768)
        # 返回裁剪后的图片和标签
        return train_image, train_label  # ndarray.uint8,HWC（image）; nddary.uint8,HW（label）
    else:  # label为空时，只裁剪image
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)  # image做的是线性插值
        # 返回裁剪后的图片
        return train_image

# color_mask中的灰度值——>对应的train_id：由给出的数据集中id和train_id的对应关系
def encode_labels(color_mask):
    # 创建和color_mask相同大小的0矩阵
    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]))  # ndarray.float64,HW
    # 根据给出的label数据集，可知train_id共有0-8，即9种类别。存在可忽略的类，即该类在训练集中没出现过，因此可以归为背景类～
    # 最后可以划分为8类，0-7；使用键值对的字典类别存储，便于后面遍历同时有对应关系～
    id_train = {
        0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218, 219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
        1: [200, 204, 209], 2: [201, 203], 3: [217], 4: [210], 5: [214],
        6: [220, 221, 222, 224, 225, 226], 7: [205, 227, 250]}  # dict类别：键值对；train_id为键（个数少），灰度值为value（个数多）
    # 完成根据value值转换成对应的键
    for i in range(8):
        # 根据键key获取对应的值value,依次遍历～
        for value in id_train[i]:
            # 在color_mask数组中为位置的位置上赋值为i
            encode_mask[color_mask == value] = i  # 赋值(将i赋值到bool矩阵中所有为true的位置)；ndarray.float64, HW
    return encode_mask  # ndarray.float64,HW

# 数据增强方法：自定义方法，要求输入是ndarray
class RandomHFlip(object):
    def __init__(self):
        super(RandomHFlip, self).__init__()
    def __call__(self, sample):
        image, label = sample
        seq = iaa.Fliplr(0.5)  # 将图片调整至原来的95.5%-110%,然后再变换成原图大小
        seg_to = seq.to_deterministic()  # 确定一个数据增强的序列seg_to，先后应用在图像和标签上面
        image = seg_to.augment_image(image)  # 应用到image上
        label = seg_to.augment_image(label)  # 应用到label上
        return image, label

class RandomVFlip(object):
    def __init__(self):
        super(RandomVFlip, self).__init__()
    def __call__(self, sample):
        image, label = sample
        seq = iaa.Flipud(0.3)  # 将图片调整至原来的95.5%-110%,然后再变换成原图大小
        seg_to = seq.to_deterministic()  # 确定一个数据增强的序列seg_to，先后应用在图像和标签上面
        image = seg_to.augment_image(image)  # 应用到image上
        label = seg_to.augment_image(label)  # 应用到label上
        return image, label

class ImageAug(object):   # 只对image做的数据增强：高斯噪声、锐化、高斯模糊
    def __call__(self, sample):
        # sample对应的是getitem中的sample，取出来对应image和label
        image, label = sample  #image为目标图片，label为对应的灰度值标签
        # 一半概率（p=0.5）进行数据增强；
        if np.random.uniform(0,1) > 0.5: # np.random.uniform(0,1)在0-1之间随机取值
            # 调整参数到还能看的出来语义信息为止，或者参考效果比较好的别人的方法，或者采取比较保守的，改变小的值～
            seq = iaa.Sequential([iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)), # 加一个高斯噪声; 噪声来自于正态分布N(L,S),loc为噪声均值，scale为噪声方差范围0~0.2*255；应用在每个通道上
                                             iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),  # 锐化
                                             iaa.GaussianBlur(sigma=(0, 1.0))])])   # 高斯模糊（高斯扰动）
            # 数据增强方法应用到image上
            image = seq.augment_image(image)
        # 返回经过增强后的图片和对应的label
        return image, label

class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size  # 指定遮挡区域大小
        self.p = p  # 随机遮挡概率
    def __call__(self, sample):
        image, label = sample  # ndarray.uint8; HWC, HW
        mask_size_half = self.mask_size // 2  # 当除2除不尽时，向下取整
        # 定义offset
        offset = 1 if self.mask_size % 2 == 0 else 0
        # 1,找到mask的中心位置(center_x,center_y)可能出现的范围
        # center_x 范围[center_x_min, center_x_max], center_y 范围[center_y_min, center_y_max]
        h, w = image.shape[:2]
        center_x_min, center_x_max = mask_size_half, w - mask_size_half + offset   # 加offset是因为下面要随机产生遮挡区域的位置，但是区间是左闭右开
        center_y_min, center_y_max = mask_size_half, h - mask_size_half + offset
        # 2，随机在中心位置所属范围随机取整数，确定中心位置
        # 当mask_size为偶数时，需要+1，防止因为右侧开，取不到；当mask_size为奇数时，因为mask_size_half是向下取整后的数，说明w-mask_size_half是少减了，即w-mask_size_half大了，所以不需要取到区间右侧数据，即不需要加1
        center_x = np.random.randint(center_x_min, center_x_max)
        center_y = np.random.randint(center_y_min, center_y_max)
        # 3,根据中心位置找到左上角的坐标和右下角的坐标（这样就能确定整个擦除框的位置）
        # 左上角的点：x_min,y_min；最小要为0
        x_min, y_min = center_x - mask_size_half, center_y - mask_size_half  # 可能出现负数
        # 右下角的点：x_max, y_max；最小要为mask_size
        x_max, y_max = x_min + self.mask_size, y_min + self.mask_size  # 根据左上角+遮挡区域长度，得到右下角；可能出现右下角坐标小于遮挡区域长度
        # 防止出现负数，同时遮挡区域不足指定尺寸，需要做限制；最大的遮挡区域左上角为(0,0),右下角为(w,h)
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
        if np.random.uniform(0, 1) < self.p:
            # 在裁剪区域填充0，显示为黑色
            image[y_min:y_max, x_min:x_max] = (0, 0, 0)
        return image, label

class ToTensor(object):  # 必须加，否则输入到网络不是tensor类型会报错
    def __init__(self):
        super(ToTensor, self).__init__()
    def __call__(self, sample):
        image, mask = sample
        # 对image进行数据类型转化
        image = np.transpose(image, (2, 0, 1))  # CHW，ndarray.uint8；过程：HWC——>CHW (因为网络要求输入格式为NCHW)
        image = image.astype(np.float32)  # CHW, ndarray.float32; 过程：numpy.uint8——>numpy.float32
        # 注意：转换数据类型为float32类型，因为torch中交叉熵损失函数要求预测图像类型为float32类型，否则要报错！！！！！！！！
        # 对label进行数据类型转化; 这一步放在计算损失函数之前更好，容易看出是为了损失函数而变成的tensor.long(int64)。。。。。。。。。。。。。
        mask = mask.astype(np.int64)  # HW(因为是灰度图), numpy.int64; 过程：numpy.float64——>numpy.int64
        # 注意：标签要搞成int64,因为torch中交叉熵损失函数要求标签类型为int64类型，否则要报错！！！！！！！！！！！！！！
        # 返回dict类型，键值对
        return {'image': torch.from_numpy(image.copy()),  # CHW, tensor.float32; 过程：numpy.float32——>tensor.float32
                'mask': torch.from_numpy(mask.copy())}    # HW, tensor.int64; 过程：numpy.int64——>tensor.int64

###################################################################################################

def get_confusion_matrix(pred, label):  # tensor.float32——>tensor.int ?
    n_class = 8
    pred, label = pred.type(torch.int), label.type(torch.int)  # 必须转化为int类型或者long类型，否则bincount报错！！！！！！！！
    # 因为输进去的pred、label本来就是tensor.int64，因为long和int都可以，但是实际上去掉后，会报错，为什么？？？
    mask = (label >= 0) & (label < n_class)  # tensor.bool, NHW (目的：使得label和pred经过掩码以后tensor变为1维tensor)
    """
    mask作用：选取正确范围内的label，拉成一维
    label[mask]  # 由label 3维（1，1710，3384）——> 1维，5786640
    pred[mask]   # 由pred  3维（1，1710，3384）——> 1维，5786640
    根据计数函数，统计混淆矩阵中对应位置（如label=7，pred=2,即7*8+2=58这个数字）出现的次数，这样的话一维展开成8x8矩阵，58这个数字对应的行索引为7，列索引为2
    """
    cm = torch.bincount(n_class * label[mask] + pred[mask], minlength=n_class ** 2)  # tenosr.int64,一维num_classes*num_classes
    """ 
    使用bincount函数统计1维tensor.int64数据中每个数出现的次数目的：
    注意：每个数代表展开成8x8矩阵时的位置，统计数出现的次数即统计位置出现的次数
    例如：统计混淆矩阵中对应位置（如label=7，pred=2,即7*8+2=58这个数字）出现的次数
    这样设计的目的是当1维展开成8x8矩阵时，58这个数字对应矩阵中位置是：行索引为7，列索引为2
    即得到 label为第7类，但是预测pred是第2类的个数
    如果minlength（代表最小长度值）被指定，那么输出数组中bin的数量至少为它指定的数。这里是8^2即64
    """
    return cm.reshape((n_class, n_class))  # 返回reshape成8x8的矩阵，confusion matrix

def get_miou(cm):
    # 混淆矩阵：放在CPU上计算
    cm = cm.cpu().numpy()   # ndarray.float32，2维(num_classes,num_classes); 过程：tensor.float32——>ndarray.float32
    # dim=0是grand truth；dim=1是pred
    iou = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))  # ndarray.float32，2维(1,num_classes)
    """
    np.sum(cm, axis=0)是各个分类的TP+FP；np.sum(cm, axis=1)是各个分类的TP+FN
    得出的iou是一个8维行向量，每个维度上的数代表一种类别的IOU交并比值
    """
    mean_iou = np.nanmean(iou)  # 原理：先求和在除
    return mean_iou

# 研讨课predict中的代码，这里使用了利用混淆矩阵计算iou的方法，因此就没用下面这种方法
def compute_iou(pred, gt, result):
    """
    pred形状 : [N, H, W]
    gt形状: [N, H, W]
    """
    # 预测结果：放在CPU上计算
    pred = pred.cpu().numpy()  # ndarray.float32, NHW（1，1710，3384）
    # 真实标签：放在CPU上计算
    gt = gt.cpu().numpy()  # ndarray.int64, NHW（1，1710，3384）
    # 一共有八种类别，对每个类别i计算TP、TP+FN、TP+FP （假设i类为正类）
    for i in range(8):
        single_gt = gt == i   # NHW, bool类型
        single_pred = pred == i  # NHW, bool类型
        # 计算iou中的分子部分
        temp_tp = np.sum(single_gt * single_pred)  # 计数，即求bool矩阵中为true的个数，即为TP的个数（i类被正确预测为i类）！！！！！
        # 计算iou中的分母部分
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        # np.sum(single_pred)是求所有预测为i类的个数；np.sum(single_gt)是求所有真实标签为i类的个数TP+FN
        # 不断累加分子和分母，用于数据迭代器循环中，当遍历完整个数据集时，最后算平均iou
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result

# 自定义交叉熵损失函数
"""
nn.CrossEntropyLoss对输入的两个变量的数据类型要求！！！！！！！！！！！！！！！！！
inputs要求的数据类型是torch.FloatTorch，即float32类型
target要求的数据类型是torch.LongTorch，即int64类型
"""
class MySoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, nbclasses):  # 指明最终的分类种数；有必要吗？？？输入到损失函数中的通道数不就是类别数吗？？？？？
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses
    def forward(self, pred, target):  # pred：tensor.float32,NCHW; target：tensor.int64,NHW
        # 改变pred维度：变成2维，通道维在最后，且为类别个数
        if pred.dim() > 2:  # N,C,H,W——>N*H*W,C
            pred = pred.view(pred.size(0), pred.size(1), -1)  # tensor.float32; N,C,H*W (view相当于reshape) 过程：N,C,H,W ——> N,C,H*W
            pred = pred.transpose(1, 2)  # transpose相当于交换维度；N,C,H*W ——> N,H*W,C 即 (1，393216，8)
            pred = pred.contiguous().view(-1, self.nbclasses)  # N,H*W,C ——> N*H*W,C 即 (393216，8)
        """
        是否可以改成下面这种写法？待验证。。
            inputs = inputs.transpose(0,2,3,1)
            inputs = inputs.contiguous().view(-1, inputs.size(2))
            # 查contiguous作用？？？？？？？？？？？？？？
        """
        # 改变label维度：变成1维
        target = target.view(-1)  # tensor.int64，变成一维(NHW)； 过程：(2,256,768) ——> 393216
        # 调用nn中的CrossEntropyLoss函数
        loss = nn.CrossEntropyLoss(reduction="mean")(pred, target)  # 会自动加上softmax
        """
            由于nn.CrossEntropuLoss对输入的参数有要求，需要先变换维度！！！！！！！！！！！！！！！！
            pred：tensor.float32, (NHW,C)二维
            label：tensor.int64,(NHW)一维
        """
        return loss

#############################################################################################################
# UNet网络：encoder + decoder 两部分！！！！！！！！！！
class ResNetUNet(nn.Module):
    # 没有bridge，是因为encoder网络使用resnet,在最后一层既改变了spatial也改变了channel，已经可以和decoder部分做直接连接了，因此没必要用bridge再去改变通道数
    def __init__(self, config):
        super(ResNetUNet, self).__init__()
        # 网络参数要跟deeplabv3p一样的参数，是同一个config
        """
        # model config
        OUTPUT_STRIDE = 16
        ASPP_OUTDIM = 256
        SHORTCUT_DIM = 48
        SHORTCUT_KERNEL = 1
        NUM_CLASSES = 8
        """
        self.n_classes = config.NUM_CLASSES
        self.padding = 1
        self.up_mode = 'upconv'  # 上采样方法用转置卷积方式
        # 这里，上采样方式仅限于转置卷积和双线性插值
        assert self.up_mode in ('upconv', 'upsample')

        # 1，encode网络：
        self.encode = ResNet101v2()  # 使用resnet101v2；返回1/4,1/8,1/16,1/32特征图
        # encoder网络采用resnet网络，其最后一层的输出通道数为2048；作为decoder网络的输入通道数（当没有bridge时）
        prev_channels = 2048

        # 2，decoder网络：前3个上采样使用转置卷积，最后1个上采样使用双线性插值～～
        # 前3次上采样网络块
        # 先定义串联decoder网络各层UNetUpBlock的空容器：up_path；也可以定义一个空的list
        self.up_path = nn.ModuleList()  # 本质还是list类型，添加要用append～
        for i in range(3):  # 循环3次UNetUpBlock（上采样——>裁剪——>融合——>双卷积基础卷积块），每次输出通道数减半
            self.up_path.append(UNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding))
            prev_channels //= 2

        # 最后一次上采样融合后，需要送到的2个3x3conv
        self.cls_conv_block1 = Block(prev_channels, 32)  # 3x3卷积+BN+relu
        self.cls_conv_block2 = Block(32, 16)  # 3x3卷积+BN+relu

        # 1x1卷积，调整输出通达为类别数
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)

        # 遍历各层，按类型分别初始化～
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  # BN层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  #x为传入的图片数据
        # blocks就是f2到f5
        input_size = x.size()[2:]
        encoder_blocks = self.encode(x)
    # 1，将encoder网络输出4个特征图中最后一个取出，作为decoder的输入
        x = encoder_blocks[-1]  # [1,2048,54,106]
        print(x.size())
        # 2，构建decoder网络中的计算逻辑：
        # 循环3次上采样基础块作为decoder网络中的前3个子模块
        for i, up_block in enumerate(self.up_path):
            print(i)
            # 对输入x先进行上采样，然后和encoder的倒数第二个特征图进行融合，然后再经过两个3x3conv～
            x = up_block(x, encoder_blocks[-i - 2])
        # 最后一个子模块：做第4次上采样，采用双线性插值，之间上采样为输入尺寸（对于resnet构成对encoder来说，这里应该是扩大4倍）
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        """
        size：表示上采样后输出的尺寸
        mode：表示上采样方法
        包括最近邻（nearest），线性插值（linear），双线性插值（bilinear），三次线性插值（trilinear），默认是最近邻（nearest）
        align_corners：设为True，在resize的时候输入图像和输出图像角点的像素将会被对齐（aligned）
        """
        # 2个3x3conv
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        # 1x1分类层
        x = self.last(x)
        return x


# decoder中的上采样基础块：上采样——>裁剪——>融合——>双卷积基础卷积块；让图像变成更高分辨率～
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding):
        super(UNetUpBlock, self).__init__()

        # 1，定义上采样层up
        # 采样转置卷积的方式进行2倍上采样
        if up_mode == 'upconv':
            self.upsampling = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)  # 通道数减半
        # 采样双线性插值的方式进行2倍上采样，需要加1x1卷积指定输出通道数～
        elif up_mode == 'upsample':
            self.upsampling = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                            nn.Conv2d(in_chans, out_chans, kernel_size=1))  # 1x1卷积指定输出通道数减半
        # UNet基础卷积块：3x3卷积+3x3卷积
        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, True)  # 使用BN层

    # 2，定义裁剪函数
    def center_crop(self, low_layer, upsampled):  # 谁大裁谁 ！！！！！！！！！！
        _, _, h1, w1 = low_layer.shape
        _, _, h2, w2 = upsampled.shape
        # 2，计算出最小的h,w
        min_h, min_w = min(h1, h2), min(w1, w2)
        # 3，计算两个图和最小值的偏差：如果大于最小值，偏差就设为两者之间的差除以2（因为采样中心裁剪方式），否则偏差就为0～
        # 中间特征图偏差
        dh1 = (h1 - min_h) // 2 if h1 > min_h else 0
        dw1 = (w1 - min_w) // 2 if w1 > min_w else 0
        # 上采样后x的偏差
        dh2 = (h2 - min_h) // 2 if h2 > min_h else 0
        dw2 = (w2 - min_w) // 2 if w2 > min_w else 0
        # 4，返回裁剪后的图
        return low_layer[:, :, dh1: (dh1 + min_h), dw1: (dw1 + min_w)], \
               upsampled[:, :, dh2: (dh2 + min_h), dw2: (dw2 + min_w)]


    def forward(self, x, low_layer):
        upsampled = self.upsampling(x)  # 上采样层，通道数减半～
        # 裁剪
        crop1, crop2 = self.center_crop(low_layer, upsampled)   #tensor.size()返回torch.Size对象，它是tuple的子类，但其使用方式与tuple略有区别
        """
        注意：如果使用bridge.size()，则返回的是一个tuple类型，upsampling.shape返回的也是tuple，
            所以center_crop函数中对bridge.size()[2:]tuple切片必须是每维分开写，不带逗号！！！！
            那么返回对切片也是tuple类型，不满足后续cat函数是对两个tensor类型进行融合返回tensor，继续输入网络中！！！
        所以，这里不使用bridge.size()，而是要用bridge（tensor类型），这样的话在center_crop函数里就可以对bridge（tensor）进行切片，当然返回的也就是tensor类型       
        """
        # 3，融合
        out = torch.cat([crop1, crop2], 1)  # 融合后，通道数翻倍～
        # 4，UNet基础卷积块：3x3卷积+3x3卷积
        out = self.conv_block(out)  # 通道数减半～
        print(out.size())
        return out

# 定义UNet基础卷积块：3x3卷积（encoder中通道数翻倍/decoder中减半）+BN+relu + 3x3卷积（通道数不变）+BN+relu
class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        """
        # 3x3卷积+relu+BN? 感觉顺序不对
        block = []
        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))
        # 修改为以下：
       """
        # 3x3卷积（通道数翻倍）+BN+relu + 3x3卷积（通道数不变）+BN+relu
        block = []
        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.ReLU())
        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))

        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


##########################################################################################
class ResNet101v2(nn.Module):
    '''
    ResNet101 model：
    1，每层残差块的个数为3，4，23，3 ;
    2，整个网络使用的残差块为带瓶颈结构的，有两种：shortcut时加1x1conv进行下采样，shortcut时不加1x1conv，即不进行下采样～～
    '''

    def __init__(self):
        super(ResNet101v2, self).__init__()
        self.conv1 = Block(3, 64, 7, 3, 2)  # 64个输出通道的7x7conv，padding=3，stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 3x3 maxpool,宽高减半
        # 第一个stage单独处理：因为只改变了channel，没改变spatial，因此stride设为1～～
        self.conv2_1 = DownBottleneck(64, 256, stride=1)  # 通道数变化：64——>256；同时shortcut时对x下采样后融合～
        self.conv2_2 = Bottleneck(256, 256)  # 通道数变化：256——>256；stride均默认为1
        self.conv2_3 = Bottleneck(256, 256)  # 通道数变化：256——>256；stride均默认为1
        """
        注意：以上3句代码不可以等价为：self.layer2 = Layer(64, [256]*2, "resnet") 
        因为Layer函数默认stride=2；而网络中的第一个stage单独处理：因为只改变了channel，没改变spatial，因此需要把stride设为1
        """
        self.layer3 = Layer(256, [512] * 4, "resnet")  # 通道数变化：256——>512,512——>512,512——>512,512——>512；stride默认为2～
        self.layer4 = Layer(512, [1024] * 23, "resnet")  # 通道数变化：512——>1024,1024——>1024,...,1024——>1024；stride默认为2～
        self.layer5 = Layer(1024, [2048] * 3, "resnet")  # 通道数变化：1024——>2048,2048——>2048,2048——>2048；stride默认为2～

    def forward(self, x):
        f1 = self.conv1(x)  # 1/2
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))  # 1/4
        f3 = self.layer3(f2)  # 1/8
        f4 = self.layer4(f3)  # 1/16
        f5 = self.layer5(f4)  # 1/32
        return [f2, f3, f4, f5]  # 返回spatial是1/4倍，1/8倍，1/16倍/，1/32倍位置的特征图～


# 定义基础组合：conv+BN+Relu，用于VGG或者resnet头部网络
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)  # inplace:选择是否进行覆盖运算,为true时可以节省内存

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

    """
    可以改写成以下写法么？？待验证～
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
    def forward(self, x):
        out = net(x)
        return out
    """


# 定义基础组合：BN+relu+conv，用于resnet～～
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


# 定义同一层内的带有瓶颈结构的残差基础块：1x1conv+3x3conv+1x1conv，stride均默认为1～
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck, self).__init__()
        assert out_chans % 4 == 0
        # BN+relu+1x1conv，通道数指定为该残差块输出通道数的1/4，为了保证后续扩大4倍后即为设定的输出通道数～～
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        # BN+relu+3x3conv，通道数不变
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        # BN+relu+1x1conv，通道数增大4倍
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        # shortcut
        out += identity  # 融合后不需要再经过激活函数了～～因为在融合前每层卷积都加了relu～～
        return out


# 定义位于不同层之间的带有瓶颈结构的残差基础块：
"""
关于stride：
适用于50层及以下的第2个stage起，第一个残差块，stride=2～
适用于50层以上每个stage的第一个残差块，但是第一个stage的第一个残差块中第一个1x1conv的stride=1；第2个stage起，第一个残差块中第一个1x1conv的stride=2～
其余stride=1
1,short时，使用1x1conv，50层以上第一个stage的第一个残差块stride=1；第2个stage起，第一个残差块stride=2～
2,整体来说输出通道数由人为指定：使用1x1conv的技巧就是，先让1x1conv输出通道数为指定该残差块输出通道数的1/4，经过3x3conv后，在用1x1conv扩大4倍，即为指定的输出通道数
"""

class DownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans,
                 stride=2):  # 默认stride=2；因为只有resnet50及以上的网络在第一个stage中spatial维不变，只改变了channel～～
        super(DownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        # BN + relu + 1x1conv，通道数指定为该残差块输出通道数的1/4，为了保证后续扩大4倍后即为设定的输出通道数～～
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=stride)
        # BN + relu + 3x3conv，通道数不变；stride=1，padding=1
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        # BN + relu + 1x1conv，通道数增大4倍
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

        # shortcut时使用的1x1conv：用于不同stage间shortcut，改变通道数和spatial维
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        identity = self.conv1(x)  # 融合前使用1x1卷积改变通道数，同时改变spatial维(h,w)大小～
        out += identity
        return out


# 构建backbone的每层，传入各个层的参数layer_list，包括in_ch, out_ch, kernel_size, padding, stride
def make_layers(in_channels, layer_list, name="vgg"):
    layers = []
    # backbone为VGG
    if name == "vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]  # 均使用3x3conv+BN+relu层；即重复堆叠，中间加上最大池化即可
            in_channels = v
    # backbone为resnet
    elif name == "resnet":
        layers += [DownBottleneck(in_channels, layer_list[0])]  # 第一个残差块使用带下采样的残差块～
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels, v)]  # 后面的残差块均使用不带下采样的残差块～
            in_channels = v  # 本层的输出通道作为下一层的输入通道～
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out


if __name__ == "__main__":
    main()