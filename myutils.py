import torch
import random
import yaml
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# 输出该文件的绝对路径除去文件名----得到项目所在的位置(OWAN-main)
PROJECT_FILE = os.path.split(os.path.realpath(__file__))[0]
CONFIG_FILE = os.path.join(PROJECT_FILE, 'configs.yml')  # 得到配置文件的路径


def get_params(param_type):
    f = open(CONFIG_FILE, encoding="utf-8")
    params = yaml.safe_load(f)  # 读取config.yml的内容
    return params[param_type]  # 根据传入的参数类型，例如AE，到路径为CONFIG_FILE的文件中寻找与AE相关的参数并返回


def TPR_FPR(y_prob, y_true, thres, verbose=True):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.where(y_prob >= thres, 1, 0)

    # Confusion matrix components
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    # === Per-class metrics ===
    # Class 1: Anomaly
    precision_1 = tp / (tp + fp + 1e-10)
    recall_1    = tp / (tp + fn + 1e-10)
    f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-10)

    # Class 0: Normal
    precision_0 = tn / (tn + fn + 1e-10)
    recall_0    = tn / (tn + fp + 1e-10)
    f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + 1e-10)

    # === Macro metrics ===
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall    = (recall_0 + recall_1) / 2
    macro_f1        = (f1_0 + f1_1) / 2

    # FPR（针对正常样本）
    fpr = fp / (fp + tn + 1e-10)

    if verbose:
        print("*********************** The relevant test indicators are as follows ***********************")
        print(f'FPR (False Positive Rate): {fpr}')
        print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
        print(f'[Anomaly Class] Precision: {precision_1}, Recall: {recall_1}, F1: {f1_1}')
        # print(f'[Normal  Class 0] Precision: {precision_0}, Recall: {recall_0}, F1: {f1_0}')
        print('--------------------')
        print(f'Macro Precision: {macro_precision}')
        print(f'Macro Recall   : {macro_recall}')
        print(f'Macro F1-Score : {macro_f1}')








# 下面函数的每个设置都是为了保证每次运行网络的时候相同输入的输出是固定的
def set_random_seed(seed=42, deterministic=True):
    random.seed(seed)  # 为随机数设定随机数种子
    np.random.seed(seed)  # 设置生成的数组的随机数种子
    torch.manual_seed(seed)  # 为cpu设置随机数种子
    torch.cuda.manual_seed_all(seed)  # 为所有的Gpu设备设置随机数种子
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
