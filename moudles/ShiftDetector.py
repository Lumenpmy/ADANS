import numpy as np
import torch
from scipy.stats import wasserstein_distance
import myutils as utils
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy.stats import entropy

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

torch.manual_seed(42)
np.random.seed(42)

class ShiftDetector:
    def __init__(self):
        pass

    def Monte_Carlo_Encoder(self, encoder,x, y, alpha=0.05, iterations=1000):
        np.random.seed(0)

        encoder.eval()

        # 提取 AE 的潜在空间表示
        with torch.no_grad():
            Z_x= encoder(torch.from_numpy(x).float())
            Z_y = encoder(torch.from_numpy(y).float())

        # print(Z_y[0].shape[0])

        # # 计算 Wasserstein 距离（按每个潜在维度单独计算，再求均值）
        # wasserstein_distances = []
        # for i in range(Z_y[0].shape[0]):
        #     w_dist = wasserstein_distance(Z_x[:, i].cpu().numpy(), Z_y[:, i].cpu().numpy())
        #     wasserstein_distances.append(w_dist)

        # observed_wasserstein_distance = sum(wasserstein_distances) / (Z_y.shape[0]) # 取平均 Wasserstein 距离
        # print("均值Wasserstein距离为:", observed_wasserstein_distance)

        def compute_SCMD(Z1, Z2):

            mu1, mu2 = Z1.mean(dim=0), Z2.mean(dim=0)  # 均值向量 (d,)
            cov1, cov2 = torch.cov(Z1.T), torch.cov(Z2.T)  # 协方差矩阵 (d, d)

            # 计算 Wasserstein-2 距离
            mean_diff = torch.norm(mu1 - mu2, p=2) ** 2  # 均值项
            cov_diff = torch.norm(cov1 - cov2, p='fro') ** 2  # 协方差项

            wasserstein_dist = mean_diff + cov_diff
            return wasserstein_dist.sqrt()

        def compute_normalized_SCMD(Z1, Z2):

            # 计算均值和标准差
            mu1, sigma1 = Z1.mean(dim=0), Z1.std(dim=0)  # 均值 & 标准差向量 (d,)
            mu2, sigma2 = Z2.mean(dim=0), Z2.std(dim=0)  # 均值 & 标准差向量 (d,)

            # 避免除以 0
            sigma1[sigma1 == 0] = 1e-6
            sigma2[sigma2 == 0] = 1e-6

            # 计算标准化后的均值项
            mean_diff = torch.norm((mu1 - mu2) / sigma1, p=2) ** 2  # 标准化均值项

            # 计算标准化后的协方差矩阵
            cov1 = torch.cov(Z1.T) / (sigma1.unsqueeze(1) @ sigma1.unsqueeze(0))  # 归一化协方差
            cov2 = torch.cov(Z2.T) / (sigma2.unsqueeze(1) @ sigma2.unsqueeze(0))  # 归一化协方差

            # 计算标准化后的协方差项
            cov_diff = torch.norm(cov1 - cov2, p='fro') ** 2  # Frobenius 范数

            SCMD= mean_diff + cov_diff
            return SCMD.sqrt()

        def compute_kl_divergence_hist(z1, z2, bins=10):
            #bins之前是50
            """
            用直方图计算潜在空间 KL 散度
            """
            hist1, bin_edges = np.histogramdd(z1, bins=bins, density=True)
            hist2, _ = np.histogramdd(z2, bins=bins, density=True)

            hist1 += 1e-10  # 避免 log(0)
            hist2 += 1e-10

            kl_div = entropy(hist1.ravel(), hist2.ravel())
            return kl_div

        kl_div_hist = compute_kl_divergence_hist(Z_x, Z_y)

        mu_x, sigma_x = Z_x.mean(dim=0), Z_x.std(dim=0)
        mu_y, sigma_y = Z_y.mean(dim=0), Z_y.std(dim=0)

        # 防止除以 0
        sigma_x[sigma_x == 0] = 1e-6
        sigma_y[sigma_y == 0] = 1e-6

        # 标准化
        Z_x_norm = (Z_x - mu_x) / sigma_x
        Z_y_norm = (Z_y - mu_y) / sigma_y

        # 计算 Wasserstein 距离（高维）
        observed_SCMD = compute_normalized_SCMD(Z_x_norm, Z_y_norm).item()
        print("SCMD为:", observed_SCMD)

        print("KL 散度（直方图）:", kl_div_hist)
        print("Z_x 均值:", Z_x.mean(dim=0))
        print("Z_y 均值:", Z_y.mean(dim=0))
        print("Z_x 方差:", Z_x.var(dim=0))
        print("Z_y 方差:", Z_y.var(dim=0))
        print("Z_x 协方差矩阵:\n", torch.cov(Z_x.T))
        print("Z_y 协方差矩阵:\n", torch.cov(Z_y.T))
        count = 0
        n = len(Z_x)
        m = len(Z_y)
        dim=y[0].shape[0]
        for i in range(iterations):
            z1_sim, z2_sim = np.random.normal(0, 1, size=(max(n, m), dim)), np.random.normal(0, 1, size=(
            max(n, m), dim))
            Z1_sim = encoder(torch.from_numpy(z1_sim).float())
            Z2_sim = encoder(torch.from_numpy(z2_sim).float())
            # 对生成的潜在空间表示进行标准化
            mu1_sim, sigma1_sim = Z1_sim.mean(dim=0), Z1_sim.std(dim=0)
            mu2_sim, sigma2_sim = Z2_sim.mean(dim=0), Z2_sim.std(dim=0)

            # 防止除以 0
            sigma1_sim[sigma1_sim == 0] = 1e-6
            sigma2_sim[sigma2_sim == 0] = 1e-6

            # 标准化
            Z1_sim_norm = (Z1_sim - mu1_sim) / sigma1_sim
            Z2_sim_norm = (Z2_sim - mu2_sim) / sigma2_sim

            # 计算模拟的 Wasserstein 距离
            simulated_stat = compute_SCMD(Z1_sim_norm, Z2_sim_norm)
            if simulated_stat >= observed_SCMD:
                count += 1
        p_value = (count + 1) / (iterations + 1)

        return p_value, observed_SCMD


    def visualize_hists(self,
                        res_1,
                        res_2,
                        color_1='#C5E0B3',
                        # F8BA63

                        color_2='#BBBBD6'):
                        #AAAAAA

        self.bin_num = utils.get_params('ShiftDetector')['test_bin_num']  # 为10
        self.bin_array = np.linspace(0., 1., self.bin_num + 1)  # 将[0,1]分成含有11个元素的均匀分布的序列
        legend_1 = 'old shifting score'  # (Calibrated)
        legend_2 = 'new shifting score'  # (Calibrated)

        res_1 = list(np.histogram(res_1, bins=self.bin_array))  # 将元组结果转成列表
        res_1[0] = res_1[0] / np.sum(res_1[0])  # cres[0]表示列表形式的每个区间的元素个数，更新后，得到频率分布的列表
        res_2 = list(np.histogram(res_2, bins=self.bin_array))
        res_2[0] = res_2[0] / np.sum(res_2[0])

        x = (res_1[1][:-1] + res_1[1][1:]) / 2
        width = x[1] - x[0]  # 区间宽度为0.1

        plt.figure('MenuBar', figsize=(10, 6))
        plt.grid(True, linewidth=1, linestyle='--')  # 网格线条

        plt.bar(x, res_1[0], width=width, alpha=0.7, ec='black', label=legend_1, color=color_1)
        plt.bar(x, res_2[0], width=width, alpha=0.5, ec='black', label=legend_2, color=color_2, hatch='//')

        def get_smooth_axis(res):
            x = (res[1][:-1] + res[1][1:]) / 2
            x = np.insert(x, 0, 0.)  # 使用数值插入函数，在x中的第0个位置插入0.
            x = np.insert(x, len(x),
                          1.)  # 最后一个插入1.,得到此时的x为[0，0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95，1]
            y = res[0]  # 表示每个区间的元素个数
            y = np.insert(y, 0, 0.)
            y = np.insert(y, len(y), 0.)
            # 进行曲线的平滑处理
            X_Y_Spline = make_interp_spline(x, y)
            X_ = np.linspace(x.min(), x.max(), 300)
            Y_ = X_Y_Spline(X_)
            return X_, Y_

        X, Y = get_smooth_axis(res_1)
        plt.plot(X, Y, '-', linewidth=8, color=color_1)
        X, Y = get_smooth_axis(res_2)
        plt.plot(X, Y, '-', linewidth=8, color=color_2)

        plt.ylim(ymin=0)
        plt.xlabel('Shifting Score', fontsize=20, fontweight='bold')  # 添加横坐标标签
        plt.ylabel('Frequency', fontsize=20, fontweight='bold')  # 添加纵坐标标签
        plt.legend(prop={'size': 20, 'weight': 'bold'})
        plt.xticks(fontsize=20, fontweight='bold')  # 增大字体并加粗
        plt.yticks(fontsize=20, fontweight='bold')  # 增大字体并加粗

        plt.show()

