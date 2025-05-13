import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import myutils as utils
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# 获取自编码器的一些参数设定
Params = utils.get_params('AE')
EvalParams = utils.get_params('Eval')  # 得到评估的参数

# torch.device：指定在哪个设备上执行模型操作
# 切换设备的操作 先判断GPU设备是否可用，如果可用则从第一个标识开始，如果不可用，则选择在cpu设备上开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()  # 常用的名词 在使用损失函数的是时候经常搭配criterion
getMSEvec = nn.MSELoss(reduction='none')


# 构造自编码器
class autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size * 0.75)),
                                     nn.ReLU(True),  # 参数inplace设置为True是为了节约内存（false为开辟内存进行计算）
                                     nn.Linear(int(feature_size * 0.75), int(feature_size * 0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.5), int(feature_size * 0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.25), int(feature_size * 0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size * 0.1), int(feature_size * 0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.25), int(feature_size * 0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.5), int(feature_size * 0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size * 0.75), int(feature_size)),
                                     )
        self.feature_size = feature_size
        self.X_train = None
        self.gradient_sorted = None

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


# reduction为none表示将每个数据和其对应的特征之间的均方误差作为向量的形式输出
# 即每个元素为（ypre-y_label)^2


# 函数的意义-----求均方根误差RMSE(RMSE和MSE的关系，RMSE=MSE^(1/2))
def se2rmse(a):
    return torch.sqrt(sum(a.t()) / a.shape[1])


# t.()是将张量a进行转置
# shape[1]表示张量a的第2维数度
# sqrt表示将括号内的东西取平方根
# 返回的结果是含有一系列数据的均方根误差的张量


def train(X_train, feature_size, epoches=Params['epoches'], lr=Params['lr']):
    # 自编码器模型进行训练，创建autoencoder的实体类model
    model = autoencoder(feature_size).to(device)  # model.to(device) 表示的是将模型加载到对应的设备上
    # Adam可以自适应的调整学习率，有助于快速的收敛
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=Params['weight_decay'])
    # 开始训练模型
    model.train()

    # 将数组类型转换为float的tensor类型
    X_train = torch.from_numpy(X_train).type(torch.float)
    if torch.cuda.is_available(): X_train = X_train.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
    # TensorDataset():进行数据的封装
    torch_dataset = Data.TensorDataset(X_train, X_train)  # x_train为何是x_train的标签？----重构学习
    # dataloader将封装好的数据进行加载
    dataloader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Params['batch_size'],
        shuffle=True,
    )
    # 进行每一轮的训练
    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            output = model(batch_x)  # 得到经过自编码器的输出值
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        if EvalParams['verbose_info']:
            print('epoch:{}/{}'.format(epoch, epoches), '|Loss:', loss.item())

    model.eval()
    output = model(X_train)
    mse_vec = getMSEvec(output, X_train)

    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()  # 得到均方根误差，并将类型转换为数组类型（因为gpu上无法进行数据的类型转换）

    if EvalParams['verbose_info']:
        print("max AD score", max(rmse_vec))

    # thres = max(rmse_vec)
    rmse_vec.sort()  # 将列表进行排列，默认为升序
    pctg = Params['percentage']
    thres = rmse_vec[int(len(rmse_vec) * pctg)]
    # ❌：将thres定义为模型经过训练输出的均方根误差的最大值
    # thres是rmse_vec的99%的那个，例如rmse_vec有200个。则thres=rmse_vec[198]
    return model, thres


@torch.no_grad()
def test(model, thres, X_test):
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)
    X_test = X_test.to(device)
    output = model(X_test)
    mse_vec = getMSEvec(output, X_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    y_pred = np.asarray([0] * len(rmse_vec))  # 生成了一个和均方根误差长度相同的元素为0的数组
    idx_mal = np.where(rmse_vec > thres)  # 找到均方根误差大于阈值的样本，并输出其位置
    # print(idx_mal)
    y_pred[idx_mal] = 1  # 使用异常置信度模型，标记异常

    return y_pred, rmse_vec


#将测试结果可视化---画散点图
def test_plot(rmse_vec, thres, file_name=None, label=None):
    # # 过滤掉 rmse_vec > 1 的样本
    # mask = rmse_vec <=1
    # rmse_vec = rmse_vec[mask]
    #
    # if label is not None:
    #     label = label[mask]  # 也要同步筛选 label

    plt.figure()
    plt.plot(np.linspace(0, len(rmse_vec) - 1, len(rmse_vec)), [thres] * len(rmse_vec), c='black',
             label='99th-threshold')

    if label is not None:
        idx = np.where(label == 0)[0]
        plt.scatter(idx, rmse_vec[idx], s=8, color='blue', alpha=0.4, label='Normal')

        idx = np.where(label == 1)[0]
        plt.scatter(idx, rmse_vec[idx], s=8, color='red', alpha=0.7, label='Abnormal')
    else:
        plt.scatter(np.linspace(0, len(rmse_vec) - 1, len(rmse_vec)), rmse_vec, s=8, alpha=0.4, label='Test samples')

    plt.legend()
    plt.xlabel('Sample NO.')
    plt.ylabel('RMSE')
    plt.title('Per-sample Score')

    if file_name is None:
        plt.show()
    else:
        plt.rcParams.update({'figure.dpi': 300})
        plt.savefig(file_name)

# def test_plot(rmse_vec, thres, file_name=None, label=None):
#     # 过滤掉 rmse_vec > 1 的样本
#     mask = rmse_vec <= 1
#     rmse_vec = rmse_vec[mask]
#
#     if label is not None:
#         label = label[mask]  # 也要同步筛选 label
#
#     fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)  # 创建两个子图
#
#     # 绘制正常样本
#     axes[0].plot(np.linspace(0, len(rmse_vec) - 1, len(rmse_vec)), [thres] * len(rmse_vec), c='black', linestyle='--',linewidth=2.5, label='99th-threshold')
#
#     if label is not None:
#         idx_normal = np.where(label == 0)[0]
#         axes[0].scatter(idx_normal, rmse_vec[idx_normal], s=8, color='blue', alpha=0.4, label='Normal')
#
#     axes[0].legend()
#     axes[0].set_ylabel('RMSE')
#     axes[0].set_title('')
#
#     # 绘制异常样本
#     axes[1].plot(np.linspace(0, len(rmse_vec) - 1, len(rmse_vec)), [thres] * len(rmse_vec), c='black', linestyle='--', label='99th-threshold')
#
#     if label is not None:
#         idx_abnormal = np.where(label == 1)[0]
#         axes[1].scatter(idx_abnormal, rmse_vec[idx_abnormal], s=8, color='red', alpha=0.7, label='Abnormal')
#
#     axes[1].legend()
#     axes[1].set_xlabel('Sample NO.')
#     axes[1].set_ylabel('RMSE')
#     axes[1].set_title('X_new_norm')
#
#     plt.tight_layout()
#
#     if file_name is None:
#         plt.savefig(r'D:\paper\1.png')
#         plt.show()
#     else:
#         plt.rcParams.update({'figure.dpi': 300})
#         plt.savefig(file_name)
