from collections import namedtuple
import random
# from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import myutils as utils

from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

#


# 获取自编码器的一些参数设定
Params = utils.get_params('AE')
EvalParams = utils.get_params('Eval')  # 得到评估的参数

# torch.device：指定在哪个设备上执行模型操作
# 切换设备的操作 先判断GPU设备是否可用，如果可用则从第一个标识开始，如果不可用，则选择在cpu设备上开始
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()  # 常用的名词 在使用损失函数的是时候经常搭配criterion
getMSEvec = nn.MSELoss(reduction='none')

try:
    AdaParams = utils.get_params('ShiftAdapter')
    ExpParams = utils.get_params('ShiftExplainer')
    EvalParams = utils.get_params('Eval')
    DetParams = utils.get_params('ShiftDetector')
except Exception as e:
    print('Error: Fail to Get Params in <configs.yml>', e)
    exit(-1)


class Screener:
    def __init__(self, model,X_o_normal, X_n, y_n, old_num, label_num, X_n_all,SCMD):
        self.X_o_normal = torch.from_numpy(X_o_normal).type(torch.float)
        self.X_n = torch.from_numpy(X_n).type(torch.float)
        self.old_num = old_num
        self.label_num = label_num
        self.y_n = y_n
        self.X_n_all = X_n_all
        self.Detector_SCMD = SCMD
        self.k = 1
        self.c = 0.5
        self.model=model
        self.p_min=0.7
        self.p_max=0.99

    def compute_alpha(self):
        """ 使用 sigmoid 函数计算 alpha """
        W_D = self.Detector_SCMD
        return 1 / (1 + np.exp(-self.k * (W_D - self.c)))

    def select_samples(self):
        """ 根据 alpha 选择样本 """
        alpha = self.compute_alpha()
        p=self.p_min+(self.p_max-self.p_min)*alpha
        # print(p)


        def encode_data(model, data_loader):
            model.eval()
            encoded_samples = []
            for batch, _ in data_loader:
                encoded = model.encoder(batch.to(device)).detach().cpu().numpy()
                encoded_samples.append(encoded)
            encoded_samples = np.vstack(encoded_samples)
            return encoded_samples

        def get_dataloader(X):

            if torch.cuda.is_available(): X = X.cuda()
            torch_dataset = Data.TensorDataset(X, X)
            dataloader = Data.DataLoader(
                dataset=torch_dataset,
                batch_size=Params['batch_size'],
                shuffle=True,
            )
            return dataloader

        old_latent_normal_representations = encode_data(self.model, get_dataloader(self.X_o_normal))

        # 计算旧数据正常样本的z的质心
        old_centroid = np.mean(old_latent_normal_representations, axis=0)

        new_latent_representations = encode_data(self.model, get_dataloader(self.X_n))
        self.old_distances = np.linalg.norm(old_latent_normal_representations - old_centroid, axis=1)
        # print(self.old_distances[:10])

        old_selected_idx = np.argsort(self.old_distances)[:self.old_num]

        # 定义旧数据的正常样本z的潜在空间便量的阈值P
        self.P = np.percentile(self.old_distances, p)

        self.new_distances = np.linalg.norm(new_latent_representations - old_centroid, axis=1)

        # print(len(self.new_distances))

        # 计算 D 值
        self.D_values = np.abs(self.new_distances - self.P)

        # 按 D 从大到小排序
        sorted_D_idx = np.argsort(self.D_values)[::-1]

        # 按 M 从大到小排序
        sorted_M_idx = np.argsort(self.new_distances)[::-1]

        # 选择远离质心的样本（D 大且 M 也大的）
        selected_far_idx = np.intersect1d(sorted_D_idx[:self.label_num], sorted_M_idx[:self.label_num])
        # print(len(selected_far_idx))

        # 选择靠近 P 的样本（D 小的）
        selected_mid_idx = sorted_D_idx[-self.label_num:]  # 从小往大选择
        # print(len(selected_mid_idx))

        # 动态调整数量
        N_far = int(alpha * self.label_num)  # 远离质心的样本数量
        N_mid = self.label_num - N_far  # 靠近 P 的样本数量
        # print(N_far, N_mid)

        # 取前 N_far 个远离质心的样本
        selected_far = selected_far_idx[:N_far]
        # 取前 N_mid 个靠近 P 的样本
        selected_mid = selected_mid_idx[:N_mid]

        new_selected_idx = np.concatenate([
            selected_far.astype(int),
            selected_mid.astype(int)
        ])


        old_rep_normal_samples = self.X_o_normal[old_selected_idx]
        new_rep_samples = self.X_n[new_selected_idx]


        new_rep_normal_samples, new_rep_normal_idx = self.HumanLabel(new_rep_samples, new_selected_idx)
        ReturnValues = namedtuple('ReturnValues',
                                  ['old_rep_normal_samples', 'old_rep_normal_idx', 'new_rep_normal_samples',
                                   'new_rep_normal_idx'])


        return ReturnValues(old_rep_normal_samples, old_selected_idx, new_rep_normal_samples, new_rep_normal_idx)

    def HumanLabel(self, new_rep_samples, new_rep_idx):
        print('NOTICE: simulating labelling...')
        remain_y_n = self.y_n[new_rep_idx]
        print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in X_i_rep')
        new_rep_normal_samples = new_rep_samples[remain_y_n == 0]
        new_rep_normal_idx = np.where(remain_y_n == 0)[0]
        print(
            " (label_num:%d, X_i_rep_normal:%d, X_i:%d)" % (self.label_num, len(new_rep_normal_samples), len(self.X_n)))
        new_rep_normal_idx = new_rep_idx[new_rep_normal_idx]
        return new_rep_normal_samples, new_rep_normal_idx



# 随机选择
class RandomSampleSelector:

    def __init__(self,model,X_o_normal, X_n,y_n,old_num,label_num,X_n_all):
        self.model = model
        self.X_o_normal=torch.from_numpy(X_o_normal).type(torch.float)
        self.X_n=torch.from_numpy(X_n).type(torch.float)
        self.old_num=old_num
        self.label_num=label_num
        self.y_n=y_n
        self.X_n_all=X_n_all

    def Random_Adapter(self):

        random_sequence_o = torch.randperm(len(self.X_o_normal))[:self.old_num]
        old_rep_samples = self.X_o_normal[random_sequence_o]
        random_sequence_n = torch.randperm(len(self.X_n))[:self.label_num]
        new_rep_samples = self.X_n[random_sequence_n]
        new_rep_normal_samples, new_rep_normal_idx = self.HumanLabel(new_rep_samples, random_sequence_n)
        return old_rep_samples,new_rep_normal_samples

    def HumanLabel(self,new_rep_samples, new_rep_idx):
        print('NOTICE: simulating labelling...')
        remain_y_n = self.y_n[new_rep_idx]
        print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in X_i_rep')
        new_rep_normal_samples = new_rep_samples[remain_y_n == 0]
        new_rep_normal_idx = np.where(remain_y_n == 0)[0]
        print(" (label_num:%d, X_i_rep_normal:%d, X_i:%d)" % ( self.label_num, len(new_rep_normal_samples), len(self.X_n)))
        new_rep_normal_idx = new_rep_idx[new_rep_normal_idx]
        return new_rep_normal_samples, new_rep_normal_idx


class UncertaintySampleSelector:

    def __init__(self,model,X_o_normal, X_n,y_n,old_num,label_num,X_n_all,X_o_rmse,X_n_rmse,thres):
        self.model = model
        self.X_o_normal=torch.from_numpy(X_o_normal).type(torch.float)
        self.X_n=torch.from_numpy(X_n).type(torch.float)
        self.old_num=old_num
        self.label_num=label_num
        self.y_n=y_n
        self.X_n_all=X_n_all
        self.X_n_rmse=X_n_rmse
        self.X_o_rmse = X_o_rmse
        self.thres=thres


    def Uncertainty(self):
        self.X_n_rmse = np.asarray(self.X_n_rmse)
        uncertainty_n = np.abs(self.X_n_rmse - self.thres)
        # 选择不确定性最高的样本（即RMSE最接近阈值的样本),因为这样的样本最需要学习
        selected_indices_n = np.argsort(uncertainty_n)[:self.label_num]
        new_rep_samples = self.X_n[selected_indices_n]


        self.X_o_rmse = np.asarray(self.X_o_rmse)
        uncertainty_o = np.abs(self.X_o_rmse - self.thres)
        # 选择不确定性最高的样本（即RMSE最接近阈值的样本),因为这样的样本最需要学习
        selected_indices_o = np.argsort(uncertainty_o)[:self.label_num]
        old_rep_normal_samples = self.X_o_normal[selected_indices_o]

        new_rep_normal_samples, new_rep_normal_idx = self.HumanLabel(new_rep_samples, selected_indices_n)

        return old_rep_normal_samples,new_rep_normal_samples

    def HumanLabel(self,new_rep_samples, new_rep_idx):
        print('NOTICE: simulating labelling...')
        remain_y_n = self.y_n[new_rep_idx]
        print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in X_i_rep')
        new_rep_normal_samples = new_rep_samples[remain_y_n == 0]
        new_rep_normal_idx = np.where(remain_y_n == 0)[0]
        print(" (label_num:%d, X_i_rep_normal:%d, X_i:%d)" % ( self.label_num, len(new_rep_normal_samples), len(self.X_n)))
        new_rep_normal_idx = new_rep_idx[new_rep_normal_idx]
        return new_rep_normal_samples, new_rep_normal_idx


class MADSampleSelector:

    def __init__(self,model,X_o_normal, X_n,y_n,old_num,label_num,X_n_all,X_n_rmse,thres):
        self.model = model # 原始paper的AEmodel需要经过对比损失训练，但是因为引用了Anoshift，所以不用对比损失
        self.X_o_normal=torch.from_numpy(X_o_normal).type(torch.float)
        self.X_n=torch.from_numpy(X_n).type(torch.float)
        self.old_num=old_num
        self.label_num=label_num
        self.y_n=y_n
        self.X_n_all=X_n_all
        self.X_n_rmse=X_n_rmse
        self.thres=thres

    def calculate_metrics_per_class(self,class_data):

        centroid = class_data.mean(dim=0)  # 形心

        # 计算每个样本到形心的欧几里得距离 d
        d = torch.norm(class_data - centroid, dim=1)

        # 计算距离 m 的中位数 n
        median = torch.median(d)

        # 计算每个样本的 |m - n| 的绝对值
        abs_diff = torch.abs(d - median)

        # 计算 |m - n| 的中位数
        abs_diff_median = torch.median(abs_diff)

        return {
            "centroid": centroid,  # 形心
            "median": median.item(),  # 距离的中位数
            "abs_diff_median": abs_diff_median.item()  # |m - n| 的中位数
        }


    def evaluate_sample_with_AE(self,sample, encoder, metrics):

        # 通过编码器得到样本的 z
        z = encoder(sample.unsqueeze(0))  # 添加 batch 维度 [1, feature_dim]

        # 计算MAD值
        MAD_values = []

        centroid = metrics["centroid"]  # 形心
        median = metrics["median"]  # 中位数
        abs_diff_median = metrics["abs_diff_median"]  # |d - median| 的中位数

        # 计算与形心的欧几里得距离 d
        d = torch.norm(z - centroid)

        MAD = torch.abs(d - median) / abs_diff_median

        # 返回最小的 MAD值和对应的类别,但是只有一类，所以是选择
        # p_min = min(MAD.item(), key=lambda x: x[0])
        return MAD
    def sort_MAD_old(self):
        # 只根据旧数据的正常样本的潜在空间得到MAD
        metrics = self.calculate_metrics_per_class(self.model.encoder(self.X_o_normal))

        # 旧数据选择距离小的
        A_old_min_list = []
        for i, sample in enumerate(self.X_o_normal):
            print(i)

            a_min = self.evaluate_sample_with_AE(sample, self.model.encoder, metrics)
            A_old_min_list.append((a_min, i))  # 保存 p_min 和对应的索引
            # 从小到大排序，并提取索引
        sorted_old_indices = [idx for _, idx in sorted(A_old_min_list, key=lambda x: x[0], reverse=False)]
        return sorted_old_indices
    def sort_MAD_new(self):
        # 只根据旧数据的正常样本的潜在空间得到MAD
        metrics = self.calculate_metrics_per_class(self.model.encoder(self.X_o_normal))

        # 新数据选择距离大的
        A_new_min_list = []
        for i, sample in enumerate(self.X_n):
            print(i)
            a_min = self.evaluate_sample_with_AE(sample, self.model.encoder, metrics)
            A_new_min_list.append((a_min, i))  # 保存 p_min 和对应的索引
            # 按  从大到小排序，并提取索引
        sorted_new_indices = [idx for _, idx in sorted(A_new_min_list, key=lambda x: x[0], reverse=True)]
        return sorted_new_indices

    def MAD(self,sorted_old_indices,sorted_new_indices):


        selected_old_indices = sorted_old_indices[:self.old_num]
        selected_new_indices = sorted_new_indices[:self.label_num]

        old_rep_samples = self.X_o_normal[selected_old_indices]
        new_rep_samples = self.X_n[selected_new_indices]
        print(len(old_rep_samples))
        print(len(new_rep_samples))
        new_rep_normal_samples, new_rep_normal_idx = self.HumanLabel(new_rep_samples, selected_new_indices)

        return old_rep_samples,new_rep_normal_samples


    def HumanLabel(self,new_rep_samples, new_rep_idx):
        print('NOTICE: simulating labelling...')
        new_rep_idx = np.array(new_rep_idx)
        remain_y_n = self.y_n[new_rep_idx]
        print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in X_i_rep')
        new_rep_normal_samples = new_rep_samples[remain_y_n == 0]
        new_rep_normal_idx = np.where(remain_y_n == 0)[0]

        print(" (label_num:%d, X_i_rep_normal:%d, X_i:%d)" % ( self.label_num, len(new_rep_normal_samples), len(self.X_n)))
        new_rep_normal_idx = new_rep_idx[new_rep_normal_idx]
        return new_rep_normal_samples, new_rep_normal_idx

