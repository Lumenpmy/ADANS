# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as Data
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance
# import AE
# import copy
# import myutils as utils
# import torch.nn.functional as F
# import os
# from torch.autograd import Function
#
#
# torch.autograd.set_detect_anomaly(True)
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# plt.switch_backend('TKAgg')
#
# try:
#     AdaParams = utils.get_params('ShiftAdapter')
# except Exception as e:
#     print('Error: Fail to Get Params in <configs.yml>', e)
#     exit(-1)
# # trp_except:用于捕获异常，try内放置可能会出现异常的句子，如果出现异常则进入exception内
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# class DANN:
#
#     def __init__(self,model,X_o_rep_nor,X_n_rep_nor,feature_size):
#         self.model = model
#         self.X_o_rep_nor=X_o_rep_nor
#         self.X_n_rep_nor = X_n_rep_nor
#         self.feature_size=feature_size
#         self.updated_encoder = None
#         self.updated_decoder = None
#         self.updated_AE = None
#     def update_encoder(self):
#
#         class GradientReversalLayer(Function):
#             @staticmethod
#             def forward(ctx, x, alpha):
#                 ctx.alpha = alpha
#                 return x.view_as(x)
#
#             @staticmethod
#             def backward(ctx, grad_output):
#                 output = grad_output.neg() * ctx.alpha
#                 return output, None
#
#         class DomainClassifier(nn.Module):
#             def __init__(self, input_size, hidden_size, dropout_rate=0.2):
#                 super(DomainClassifier, self).__init__()
#                 self.fc1 = nn.Linear(input_size, hidden_size)
#                 self.fc2 = nn.Linear(hidden_size, hidden_size)
#                 self.fc3 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点
#                 self.dropout = nn.Dropout(dropout_rate)
#
#             def forward(self, x):
#                 x = F.relu(self.fc1(x))
#                 x = self.dropout(x)
#                 x = F.relu(self.fc2(x))
#                 x = F.relu(self.fc2(x))
#                 x = self.dropout(x)
#                 weights = (self.fc3(x))  # 使用 sigmoid 将输出限制在 0 到 1 之间，表示样本权重
#                 return weights
#
#         class LabelPredictor(nn.Module):
#             def __init__(self, input_size, hidden_size,dropout_rate=0.1):
#                 super(LabelPredictor, self).__init__()
#                 self.fc1 = nn.Linear(input_size, hidden_size)
#                 self.fc2 = nn.Linear(hidden_size, hidden_size)
#                 self.fc3 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点
#                 self.dropout = nn.Dropout(dropout_rate)
#
#             def forward(self, x):
#                 x = F.relu(self.fc1(x))
#                 x = self.dropout(x)
#                 x = F.relu(self.fc2(x))
#                 x = self.dropout(x)
#                 x = (self.fc3(x))
#                 return x
#
#
#
#         def train_domain_adversarial(feature_extractor, domain_classifier,label_predictor,data_loader_A, data_loader_B, optimizer_classify,
#                                      optimizer_confuse,optimizer_predictor,criterion_domain,criterion_label,  num_epoches, weight_adjust_factor_A, weight_adjust_factor_B):
#             def adapt_alpha(current_epoch, max_epoch):
#                 return 0.5*(current_epoch / max_epoch)
#             def grad_reverse(x, alpha):
#                 return GradientReversalLayer.apply(x, alpha)
#
#             def print_gradients(model):
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         grad_data = param.grad.data
#                         print(
#                             f'Layer: {name} | Grad Max: {grad_data.max()} | Grad Min: {grad_data.min()} | Grad Mean: {grad_data.mean()}')
#                     else:
#                         print(f'Layer: {name} has no gradient')
#
#             domain_weights_A_all = []
#             domain_weights_B_all = []
#
#             for epoch in range(num_epoches):
#                 alpha = adapt_alpha(epoch, num_epoches)
#                 for data_A, data_B in zip(data_loader_A, data_loader_B):
#
#                     inputs_A, _ = data_A
#                     inputs_B, _ = data_B
#                     features_A = feature_extractor(inputs_A)
#                     features_B = feature_extractor(inputs_B)
#                     reversed_features_A = grad_reverse(features_A, alpha)
#                     reversed_features_B = grad_reverse(features_B, alpha)
#
#                     domain_weights_A = domain_classifier(reversed_features_A)
#                     domain_weights_B = domain_classifier(reversed_features_B)
#
#                     # 根据权重调整对抗训练损失函数
#                     classifier_loss_A = criterion_domain(domain_weights_A, torch.zeros((len(inputs_A), 1))) * weight_adjust_factor_A  # 使用权重调整因子调整样本A的损失
#                     classifier_loss_B  = criterion_domain(domain_weights_B, torch.ones((len(inputs_B), 1))) * weight_adjust_factor_B
#                     classifier_loss=classifier_loss_A  + classifier_loss_B
#                     optimizer_classify.zero_grad()
#                     optimizer_confuse.zero_grad()
#                     classifier_loss.backward()
#                     # print_gradients(domain_classifier)
#                     optimizer_classify.step()
#                     optimizer_confuse.step()
#
#                     optimizer_predictor.zero_grad()
#                     label_output_A = label_predictor(features_A.detach())
#                     predict_loss=criterion_label(label_output_A,inputs_A)
#                     predict_loss.backward()
#                     # print_gradients(label_predictor)
#                     optimizer_predictor.step()
#
#                     if (epoch == num_epoches - 1):
#                         domain_weights_A_all.append(domain_weights_A.detach().cpu().numpy())
#                         domain_weights_B_all.append(domain_weights_B.detach().cpu().numpy())
#                 print('epoch:{}/{}'.format(epoch, num_epoches),"old_domain_classfiy_loss:",(classifier_loss_A).item(),"new_domain_classfiy_loss:",(classifier_loss_B).item(),"domain_classfiy_loss:",(classifier_loss_A+classifier_loss_B).item(),"label_predictor_loss",predict_loss.item())
#                 # if (epoch == num_epoches - 1):
#                 #     domain_weights_A_all = np.concatenate(domain_weights_A_all)
#                 #     domain_weights_B_all = np.concatenate(domain_weights_B_all)
#
#             # return {"domain_weights_A": domain_weights_A_all, "domain_weights_B": domain_weights_B_all}
#             return feature_extractor
#         # 实例化自编码器M和领域分类器
#         num_epoches = 20
#         learning_rate_d = 0.01
#         learning_rate_f=0.01
#         learning_rate_l=0.01
#         criterion_domain = nn.BCEWithLogitsLoss()
#         criterion_label = nn.MSELoss()
#
#         feature_extractor=copy.deepcopy(self.model.encoder)
#         domain_classifier = DomainClassifier(input_size=int(self.feature_size * 0.1), hidden_size=100)
#         label_predictor=copy.deepcopy(self.model.decoder)
#         optimizer_classify = optim.Adam(list(domain_classifier.parameters()),lr=learning_rate_d)
#         optimizer_confuse = optim.Adam(list(feature_extractor.parameters()), lr=learning_rate_f)
#         optimizer_predictor = optim.Adam(list(label_predictor.parameters()), lr=learning_rate_l)
#
#         # TensorDataset():进行数据的封装
#         # X_o_rep_nor = torch.from_numpy(self.X_o_rep_nor).type(torch.float)
#         if torch.cuda.is_available(): X_o_rep_nor = self.X_o_rep_nor.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
#         # TensorDataset():进行数据的封装
#         X_o_rep_nor=self.X_o_rep_nor
#
#         torch_dataset = Data.TensorDataset(X_o_rep_nor, X_o_rep_nor)  # x_train为何是x_train的标签？----重构学习
#         # dataloader将封装好的数据进行加载
#         data_loader_A = Data.DataLoader(
#             dataset=torch_dataset,
#             batch_size=1024,
#             shuffle=True,
#         )
#
#         # X_n_rep_nor = torch.from_numpy(self.X_n_rep_nor).type(torch.float)
#         X_n_rep_nor = self.X_n_rep_nor
#         if torch.cuda.is_available(): X_n_rep_nor = self.X_n_rep_nor.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
#         torch_dataset = Data.TensorDataset(X_n_rep_nor, X_n_rep_nor)  # x_train为何是x_train的标签？----重构学习
#         # dataloader将封装好的数据进行加载
#         data_loader_B = Data.DataLoader(
#             dataset=torch_dataset,
#             batch_size=1024,
#             shuffle=True,
#         )
#         self.updated_encoder=train_domain_adversarial(feature_extractor, domain_classifier,label_predictor,data_loader_A,
#                                           data_loader_B, optimizer_classify,optimizer_confuse,optimizer_predictor,
#                                           criterion_domain,criterion_label,num_epoches,0.5, 0.5)
#     def update_decoder(self):
#         class Discriminator(nn.Module):
#             def __init__(self, input_size, hidden_size):
#                 super(Discriminator, self).__init__()
#                 self.fc1 = nn.Linear(input_size, hidden_size)
#                 self.fc2 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点
#
#             def forward(self, x):
#                 x = F.relu(self.fc1(x))
#                 x = (self.fc2(x))  # 使用 sigmoid 将输出限制在 0 到 1 之间，表示样本权重
#                 return x
#
#         num_epoches = 10
#         learning_rate_d = 0.0002
#         learning_rate_g = 0.001
#         # criterion = nn.BCELoss()
#         criterion = nn.BCEWithLogitsLoss()
#
#         generator = copy.deepcopy(self.model.decoder)
#         discriminator = Discriminator(input_size=self.feature_size, hidden_size=int(self.feature_size * 0.1))
#         optimizer_generator = optim.Adam(list(generator.parameters()), lr=learning_rate_g)
#         optimizer_discriminator = optim.RMSprop(list(discriminator.parameters()), lr=learning_rate_d)
#
#         self.X_rep_nor = torch.cat((self.X_o_rep_nor, self.X_n_rep_nor), dim=0)
#         if torch.cuda.is_available(): self.X_rep_nor = self.X_rep_nor.cuda()  # .cuda()函数：表示将数据拷贝在GPU上
#         # TensorDataset():进行数据的封装
#         torch_dataset = Data.TensorDataset(self.X_rep_nor, self.X_rep_nor)  # x_train为何是x_train的标签？----重构学习
#         # dataloader将封装好的数据进行加载
#         data_loader = Data.DataLoader(
#             dataset=torch_dataset,
#             batch_size=1024,
#             shuffle=True,
#         )
#
#         for epoch in range(num_epoches):
#             for step, (batch_x, batch_y) in enumerate(data_loader):
#                 feature = self.updated_encoder(batch_x)
#                 fake = generator(feature)
#
#                 real_labels = torch.full((batch_x.size(0), 1), 0.9).cuda() if torch.cuda.is_available() else torch.full(
#                     (batch_x.size(0), 1), 0.9)
#                 fake_labels = torch.zeros((batch_x.size(0), 1)).cuda() if torch.cuda.is_available() else torch.zeros(
#                     (batch_x.size(0), 1))
#
#                 # 计算真实数据和假数据的鉴别器损失
#                 logits_real = discriminator(batch_x)
#                 logits_fake = discriminator(fake.detach())  # 假数据需要detach以避免计算图分离
#                 # d_loss_real = criterion(logits_real, torch.ones_like(logits_real))
#                 d_loss_real = criterion(logits_real, real_labels )
#                 # d_loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
#                 d_loss_fake = criterion(logits_fake, fake_labels)
#                 d_loss = d_loss_real + d_loss_fake
#
#                 # 优化鉴别器
#                 optimizer_discriminator.zero_grad()
#                 d_loss.backward()
#                 optimizer_discriminator.step()
#
#                 logits_fake = discriminator(fake)  # 重新计算假数据的判别结果
#                 g_loss = criterion(logits_fake, torch.ones_like(logits_fake))
#                 optimizer_generator.zero_grad()
#                 g_loss.backward()
#                 optimizer_generator.step()
#
#             print(
#                 f'Epoch [{epoch}/{num_epoches}], real_discriminator_loss: {d_loss_real.item()}, fake_discriminator_loss: {d_loss_fake.item()},generator_loss: {g_loss.item()}')
#         self.updated_decoder=generator
#
#     def update_AE(self):
#         class Autoencoder(nn.Module):
#             def __init__(self, encoder, decoder):
#                 super(Autoencoder, self).__init__()
#                 self.encoder = encoder
#                 self.decoder = decoder
#
#             def forward(self, x):
#                 encoded = self.encoder(x)
#                 decoded = self.decoder(encoded)
#                 return decoded
#         self.update_encoder()
#         self.update_decoder()
#
#         self.updated_AE= Autoencoder(self.updated_encoder, self.updated_decoder)


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


class DANN:

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
            # def forward(ctx, x, alpha):
            #     ctx.alpha = alpha
            #     return x.view_as(x)
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.clone()

            @staticmethod
            # def backward(ctx, grad_output):
            #     output = grad_output.neg() * ctx.alpha
            #     return output, None
            def backward(ctx, grad_output):
                return -ctx.alpha * grad_output, None  # 反向传播时翻转梯度

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
                x = self.dropout(x)
                return self.fc3(x)  # 不要加 sigmoid

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
            # def adapt_alpha(current_epoch, max_epoch):
            #     return 0.5 * (current_epoch / max_epoch)

            def adapt_alpha(current_epoch, max_epoch, gamma=10):# GRL层的梯度缩放因子采用指数策略，比起上面的，后期趋近于1，对抗效果更强
                return 2 / (1 + np.exp(-gamma * (current_epoch / max_epoch))) - 1

            def grad_reverse(x, alpha):
                return GradientReversalLayer.apply(x, alpha)

            def se2rmse(a):
                return torch.sqrt(sum(a.t()) / a.shape[1])

            def print_gradients(model, name):
                print(f"\nBefore backward, gradients for {name}:")
                for param_name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"No gradient for {param_name}")
                    else:
                        print(f" {param_name} | Mean Grad: {param.grad.mean().item():.6f}")


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

                    classifier_loss.backward(retain_graph=True)
                    # print_gradients(domain_classifier, "domain classifier")  # 确保梯度正常
                    optimizer_classifier.step()
                    # optimizer_encoder.step()

                    # print(f"features_A.requires_grad: {features_A.requires_grad}")
                    # print(f"features_A.grad_fn: {features_A.grad_fn}")  # 这个应当不为 None

                    reconstruct_data_A = AE_decoder(features_A)
                    reconstruct_data_B = AE_decoder(features_B)

                    # print(f"label_predictor output shape: {reconstruct_data_A.shape}")
                    # print(f"label_predictor output requires_grad: {reconstruct_data_A.requires_grad}")
                    # print(f"label_predictor output.grad_fn: {reconstruct_data_A.grad_fn}")  # 这个应当不为 None


                    mse_loss = nn.MSELoss(reduction='none')
                    loss_A = mse_loss(reconstruct_data_A, inputs_A)
                    rmse_vec_A = torch.sqrt(loss_A.mean(dim=1))
                    # if(rmse_vec_A>thres):
                    #     predict_loss=rmse_vec_A
                    # else:predict_loss=0
                    predict_loss_A = torch.where(rmse_vec_A < self.thres, torch.zeros_like(rmse_vec_A), rmse_vec_A)
                    predict_loss = predict_loss_A.mean()

                    loss_B = mse_loss(reconstruct_data_B, inputs_B)
                    # rmse_vec_B = se2rmse(loss_B).cpu().data.numpy() * weight_adjust_factor_B

                    # rmse_vec_A = torch.sqrt(loss_A.mean(dim=1)) * weight_adjust_factor_A

                    # rmse_vec_B = torch.sqrt(loss_B.mean(dim=1)) * weight_adjust_factor_B

                    # 根据阈值处理 RMSE

                    # predict_loss_B = torch.where(rmse_vec_B < self.thres, torch.zeros_like(rmse_vec_B), rmse_vec_B)

                    # predict_loss = predict_loss_A.mean() + predict_loss_B.mean()
                    optimizer_decoder.zero_grad()
                    # optimizer_encoder.zero_grad()
                    # print(f"predict_loss: {predict_loss.item()}")
                    predict_loss.backward()
                    # print_gradients(AE_encoder, "label predictor")  # 确保梯度正常

                    optimizer_decoder.step()
                    optimizer_encoder.step()

                print('epoch:{}/{}'.format(epoch, num_epoches)),
                print("old_domain_classfiy_loss:", (classifier_loss_A).item(),
                      "new_domain_classfiy_loss:",(classifier_loss_B).item(),
                      "domain_classfiy_loss:",(classifier_loss_A + classifier_loss_B).item())
                print("label_predictor_loss:",(predict_loss).item())
                print("feature_extractor_loss:",(classifier_loss_A + classifier_loss_B+predict_loss).item())
                # print("AE-encoder_loss",(classifier_loss + predict_loss).item())

            # return {"domain_weights_A": domain_weights_A_all, "domain_weights_B": domain_weights_B_all}
            return AE_encoder, AE_decoder

        # 实例化自编码器M和领域分类器
        if (self.labeling_probability == 0.01):
            num_epoches = 20
            learning_rate_classifier = 0.01
            learning_rate_encoder = 0.01
            learning_rate_decoder = 0.09
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5
        if (self.labeling_probability == 0.05):
            num_epoches = 20
            learning_rate_classifier = 0.01
            learning_rate_encoder = 0.01
            learning_rate_decoder = 0.1
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5
        if(self.labeling_probability==0.1):
            num_epoches = 20
            learning_rate_classifier = 0.009
            learning_rate_encoder = 0.01
            learning_rate_decoder = 0.09
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5
        if(self.labeling_probability==0.2):
            num_epoches = 20
            learning_rate_classifier = 0.009
            learning_rate_encoder = 0.01
            learning_rate_decoder = 0.09
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5
        if (self.labeling_probability == 0.3):
            num_epoches = 20
            learning_rate_classifier = 0.009
            learning_rate_encoder = 0.01
            learning_rate_decoder = 0.09
            weight_adjust_factor_A = 0.5
            weight_adjust_factor_B = 0.5


        criterion_mse = nn.BCEWithLogitsLoss()

        AE_encoder = copy.deepcopy(self.model.encoder)
        for param in AE_encoder.parameters():
            param.requires_grad = True  # 确保梯度计算
        optimizer_encoder = optim.Adam(list(AE_encoder.parameters()), lr=learning_rate_encoder)

        domain_classifier = DomainClassifier(input_size=int(self.feature_size * 0.1), hidden_size=100)
        for param in domain_classifier.parameters():
            param.requires_grad = True
        optimizer_classifier = optim.Adam(list(domain_classifier.parameters()), lr=learning_rate_classifier)

        AE_decoder = copy.deepcopy(self.model.decoder)
        for param in AE_decoder.parameters():
            param.requires_grad = True
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

    def change_thres(self, model, x):
        def se2rmse(a):
            return torch.sqrt(sum(a.t()) / a.shape[1])

        model.eval()
        output = model(x)
        getMSEvec = nn.MSELoss(reduction='none')
        mse_vec = getMSEvec(output, x)
        rmse_vec = se2rmse(mse_vec).cpu().data.numpy()  # 得到均方根误差，并将类型转换为数组类型（因为gpu上无法进行数据的类型转换）
        rmse_vec.sort()  # 将列表进行排列，默认为升序
        pctg = 0.99
        thres = rmse_vec[int(len(rmse_vec) * pctg)]
        return thres













