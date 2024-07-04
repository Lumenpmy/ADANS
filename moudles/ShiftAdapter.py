
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import AE
import copy
import myutils as utils
import torch.nn.functional as F
import os
from torch.autograd import Function


torch.autograd.set_detect_anomaly(True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.switch_backend('TKAgg')

try:
    AdaParams = utils.get_params('ShiftAdapter')
    ExpParams = utils.get_params('ShiftExplainer')
    EvalParams = utils.get_params('Eval')
except Exception as e:
    print('Error: Fail to Get Params in <configs.yml>', e)
    exit(-1)
# trp_except:用于捕获异常，try内放置可能会出现异常的句子，如果出现异常则进入exception内

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ShiftAdapter:

    def __init__(self, model, X_o_rep_nor, X_n_rep_nor, feature_size, thres,labeling_probability):
        self.model = model
        self.X_o_rep_nor = X_o_rep_nor
        self.X_n_rep_nor = X_n_rep_nor
        self.feature_size = feature_size
        self.updated_encoder = None
        self.updated_decoder = None
        self.updated_AE = None
        self.thres = thres
        self.labeling_probability=labeling_probability

    def update(self):

        class GradientReversalLayer(Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                output = grad_output.neg() * ctx.alpha
                return output, None

        class DomainClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_rate=0.2):
                super(DomainClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                weights = (self.fc3(x))  # 使用 sigmoid 将输出限制在 0 到 1 之间，表示样本权重
                return weights

        class Discriminator(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(Discriminator, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = (self.fc2(x))  # 使用 sigmoid 将输出限制在 0 到 1 之间，表示样本权重
                return x

        def train_domain_adversarial(AE_encoder, domain_classifier, AE_decoder, data_loader_A,
                                     data_loader_B, optimizer_encoder, optimizer_classifier,
                                     optimizer_decoder,criterion_mse, num_epoches, weight_adjust_factor_A,weight_adjust_factor_B):
            def adapt_alpha(current_epoch, max_epoch):
                return 0.5 * (current_epoch / max_epoch)

            def grad_reverse(x, alpha):
                return GradientReversalLayer.apply(x, alpha)

            def se2rmse(a):
                return torch.sqrt(sum(a.t()) / a.shape[1])

            def print_gradients(model):
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_data = param.grad.data
                        print(
                            f'Layer: {name} | Grad Max: {grad_data.max()} | Grad Min: {grad_data.min()} | Grad Mean: {grad_data.mean()}')
                    else:
                        print(f'Layer: {name} has no gradient')



            for epoch in range(num_epoches):
                alpha = adapt_alpha(epoch, num_epoches)
                for data_A, data_B in zip(data_loader_A, data_loader_B):

                    inputs_A, _ = data_A
                    inputs_B, _ = data_B
                    features_A = AE_encoder(inputs_A)
                    features_B = AE_encoder(inputs_B)
                    reversed_features_A = grad_reverse(features_A, alpha)
                    reversed_features_B = grad_reverse(features_B, alpha)

                    domain_weights_A = domain_classifier(reversed_features_A)
                    domain_weights_B = domain_classifier(reversed_features_B)

                    # 根据权重调整对抗训练损失函数
                    classifier_loss_A = criterion_mse(domain_weights_A, torch.zeros(
                        (len(inputs_A), 1))) * weight_adjust_factor_A  # 使用权重调整因子调整样本A的损失
                    classifier_loss_B = criterion_mse(domain_weights_B,
                                                      torch.ones((len(inputs_B), 1))) * weight_adjust_factor_B
                    classifier_loss = classifier_loss_A + classifier_loss_B
                    optimizer_classifier.zero_grad()
                    optimizer_encoder.zero_grad()

                    classifier_loss.backward()
                    # print_gradients(domain_classifier)
                    optimizer_classifier.step()
                    optimizer_encoder.step()

                    optimizer_decoder.zero_grad()
                    optimizer_encoder.zero_grad()
                    reconstruct_data_A = AE_decoder(features_A.detach())
                    reconstruct_data_B = AE_decoder(features_B.detach())


                    mse_loss = nn.MSELoss(reduction='none')
                    loss_A = mse_loss(reconstruct_data_A, inputs_A)
                    rmse_vec_A = torch.sqrt(loss_A.mean(dim=1))
                    predict_loss_A = torch.where(rmse_vec_A < self.thres, torch.zeros_like(rmse_vec_A), rmse_vec_A)
                    predict_loss = predict_loss_A.mean()
                    loss_B = mse_loss(reconstruct_data_B, inputs_B)
                    predict_loss.backward()
                    optimizer_decoder.step()
                    optimizer_encoder.step()

                print('epoch:{}/{}'.format(epoch, num_epoches)),
                print("old_domain_classfiy_loss:", (classifier_loss_A).item(),
                      "new_domain_classfiy_loss:",(classifier_loss_B).item(),
                      "domain_classfiy_loss:",(classifier_loss_A + classifier_loss_B).item())
                print("label_predictor_loss",(predict_loss).item())

            return AE_encoder, AE_decoder

        if (self.labeling_probability == 0.01):
            num_epoches = 20
            learning_rate_classifier = 0.051
            learning_rate_encoder = 0.049
            learning_rate_decoder = 0.045
            weight_adjust_factor_A=0.5
            weight_adjust_factor_B=0.5
        if(self.labeling_probability==0.1):
            num_epoches = 20
            learning_rate_classifier = 0.021
            learning_rate_encoder = 0.021
            learning_rate_decoder = 0.00179
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5
        if(self.labeling_probability==0.2):
            num_epoches = 20
            learning_rate_classifier = 0.05
            learning_rate_encoder = 0.05
            learning_rate_decoder = 0.001
            weight_adjust_factor_A = 0.49
            weight_adjust_factor_B = 0.51
        if (self.labeling_probability == 0.3):
            num_epoches = 20
            learning_rate_classifier = 0.05
            learning_rate_encoder = 0.05
            learning_rate_decoder = 0.0005
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5
        if (self.labeling_probability == 0.4):
            num_epoches = 20
            learning_rate_classifier = 0.05
            learning_rate_encoder = 0.05
            learning_rate_decoder = 0.0005
            weight_adjust_factor_A=0.5
            weight_adjust_factor_B=0.5
        if (self.labeling_probability == 0.5):
            num_epoches = 20
            learning_rate_classifier = 0.1
            learning_rate_encoder = 0.05
            learning_rate_decoder = 0.00045
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5

        criterion_mse = nn.BCEWithLogitsLoss()

        AE_encoder = copy.deepcopy(self.model.encoder)
        optimizer_encoder = optim.Adam(list(AE_encoder.parameters()), lr=learning_rate_encoder)

        domain_classifier = DomainClassifier(input_size=int(self.feature_size * 0.1), hidden_size=100)
        optimizer_classifier = optim.Adam(list(domain_classifier.parameters()), lr=learning_rate_classifier)

        AE_decoder = copy.deepcopy(self.model.decoder)
        optimizer_decoder = optim.Adam(list(AE_decoder.parameters()), lr=learning_rate_decoder)

        X_o_rep_nor = self.X_o_rep_nor
        if torch.cuda.is_available(): X_o_rep_nor = self.X_o_rep_nor.cuda()

        torch_dataset = Data.TensorDataset(X_o_rep_nor, X_o_rep_nor)
        data_loader_A = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=1024,
            shuffle=True,
        )

        X_n_rep_nor = self.X_n_rep_nor
        if torch.cuda.is_available(): X_n_rep_nor = self.X_n_rep_nor.cuda()  # .cuda()函数：表示将数据拷贝在GPU上

        torch_dataset = Data.TensorDataset(X_n_rep_nor, X_n_rep_nor)  # x_train为何是x_train的标签？----重构学习
        # dataloader将封装好的数据进行加载
        data_loader_B = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=1024,
            shuffle=True,
        )

        self.updated_encoder, self.updated_decoder = train_domain_adversarial(AE_encoder, domain_classifier, AE_decoder,data_loader_A,
                                                                              data_loader_B, optimizer_encoder,optimizer_classifier,
                                                                              optimizer_decoder,criterion_mse,num_epoches, weight_adjust_factor_A,weight_adjust_factor_B)

    def update_AE(self):
        class Autoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super(Autoencoder, self).__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        self.update()

        self.updated_AE = Autoencoder(self.updated_encoder, self.updated_decoder)













