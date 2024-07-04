from collections import namedtuple
import random
from scipy.stats import wasserstein_distance

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

class ShiftScreener:

    def __init__(self,model,X_o_normal, X_n,y_n,old_num,label_num,X_n_all):
        self.model = model
        self.X_o_normal=torch.from_numpy(X_o_normal).type(torch.float)
        self.X_n=torch.from_numpy(X_n).type(torch.float)
        self.old_num=old_num
        self.label_num=label_num
        self.y_n=y_n
        self.X_n_all=X_n_all

    def PSA(self):

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

        def tsne_reduce_and_plot(latent_space, representative_indices=None):
            tsne = TSNE(n_components=2, random_state=0)
            tsne_results = tsne.fit_transform(latent_space)

            plt.figure(figsize=(8, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='#C5E0B3', alpha=0.5)

            if representative_indices is not None:
                plt.scatter(tsne_results[representative_indices, 0], tsne_results[representative_indices, 1], c='red')



            plt.show()

        # 对筛选后的正常样本进行编码
        old_latent_normal_representations = encode_data(self.model, get_dataloader(self.X_o_normal))
        # 对正常样本进行K-means聚类，这里n_clusters=1，因为只对正常样本聚类
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=0).fit(old_latent_normal_representations)
        # 计算所有正常样本到聚类中心的距离
        closest_indices, distances_o_normal = pairwise_distances_argmin_min(old_latent_normal_representations,
                                                                   kmeans.cluster_centers_)
        #tsne_reduce_and_plot(old_latent_normal_representations, representative_indices=closest_indices)
        #title = 't-SNE plot with Hidden Variable-KMeans Cluster Center'
        new_latent_representations = encode_data(self.model, get_dataloader(self.X_n))
        distances_n = np.linalg.norm(new_latent_representations - kmeans.cluster_centers_[0], axis=1)
        return distances_o_normal,distances_n

    def se2rmse(self,a):
        return torch.sqrt(sum(a.t()) / a.shape[1])

    def RMSE_Kmeans(self):

        with torch.no_grad():
            output_o_normal = self.model(self.X_o_normal)

        mse_loss = nn.MSELoss(reduction='none')
        mse_vec_o_normal = mse_loss(output_o_normal, self.X_o_normal)
        rmse_o_normal = self.se2rmse(mse_vec_o_normal).cpu().data.numpy()
        rmse_o_normal = rmse_o_normal.reshape(-1, 1)

        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(rmse_o_normal)
        cluster_centers = kmeans.cluster_centers_
        closest, rmse_distances_o_normal = pairwise_distances_argmin_min(rmse_o_normal, cluster_centers)

        with torch.no_grad():
            output_n = self.model(self.X_n)
        mse_vec_n = mse_loss(output_n, self.X_n)
        rmse_n = self.se2rmse(mse_vec_n).cpu().data.numpy()  # 得到均方根误差，并将类型转换为数组类型（因为gpu上无法进行数据的类型转换）
        rmse_n = rmse_n.reshape(-1, 1)
        closest, rmse_distances_n = pairwise_distances_argmin_min(rmse_n, cluster_centers)

        def plot_rmse_kmeans_results(rmse_o_normal, rmse_n, cluster_centers):
            plt.figure(figsize=(4, 1))

            # 绘制旧数据的真正样本的RMSE围绕聚类中心
            plt.scatter(rmse_o_normal, [0] * len(rmse_o_normal), color='#C5E0B3', alpha=0.6)

            # 绘制聚类中心
            plt.scatter(cluster_centers, [0], color='red', marker='x',)

            plt.gca().axes.get_yaxis().set_visible(False)

            # plt.legend()
            plt.show()
        # 假设 rmse_o_normal, rmse_n 和 cluster_centers 是你的数据

        distances_feature_o_normal, distances_feature_n=self.PSA()
        score_o_normal=distances_feature_o_normal*rmse_distances_o_normal
        score_n=distances_feature_n*rmse_distances_n

        old_rep_normal_idx = np.argsort(score_o_normal)[: self.old_num]
        old_rep_normal_samples = self.X_o_normal[old_rep_normal_idx]

        new_rep_idx = np.argsort(-score_n)[:self.label_num]
        new_rep_samples = self.X_n[new_rep_idx]

        new_rep_normal_samples, new_rep_normal_idx=self.HumanLabel(new_rep_samples,new_rep_idx)
        ReturnValues = namedtuple('ReturnValues', ['old_rep_normal_samples', 'old_rep_normal_idx','new_rep_normal_samples','new_rep_normal_idx'])

        mixed_distribution = torch.cat((old_rep_normal_samples, new_rep_normal_samples), dim=0)
        # print('Remain_X_o.shape', top_samples_A.shape, 'Remain X_n.shape', top_samples_B.shape)

        # 计算这些样本之间的Wasserstein距离

        # random_sequence_n = random.sample(range(0, len(self.X_n_all)), len(mixed_distribution))
        # w_dist = wasserstein_distance(mixed_distribution.ravel(), self.X_n_all[random_sequence_n].ravel())
        # print("Wasserstein distance between the two distributions is: ", w_dist)
        return ReturnValues(old_rep_normal_samples,old_rep_normal_idx,new_rep_normal_samples,new_rep_normal_idx)

    def HumanLabel(self,new_rep_samples, new_rep_idx):
        print('NOTICE: simulating labelling...')
        remain_y_n = self.y_n[new_rep_idx]
        print('Filter', len(remain_y_n[remain_y_n == 1]), 'anomalies in X_n_rep')
        new_rep_normal_samples = new_rep_samples[remain_y_n == 0]
        new_rep_normal_idx = np.where(remain_y_n == 0)[0]
        print(" (label_num:%d, X_n_rep_normal:%d, X_n:%d)" % ( self.label_num, len(new_rep_normal_samples), len(self.X_n)))
        new_rep_normal_idx = new_rep_idx[new_rep_normal_idx]
        return new_rep_normal_samples, new_rep_normal_idx

