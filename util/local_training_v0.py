# util/local_training_v0.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np 
# 导入 v0 的 get_output
from .util_v0 import get_output 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # 论文 [cite: 495] 中使用交叉熵损失 l(x_i, y_i; w)
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs

    def update_weights(self, net, seed, w_g, epoch, mu=0): 
        # mu 在 v0 中不使用 (mu 用于 FedProx)
        
        net.train()
        # 论文 [cite: 529] 中提到基于 SGD
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        # 论文 [cite: 546] 中的 E (local_ep)
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device).long()
                
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                
                # (移除了 beta FedProx 项)
                # (移除了 mixup 逻辑)
                # (移除了 APCL 对比损失)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # (移除了 calculate_sub_prototypes 函数)
    
def globaltest(net, test_dataset, args):
    # (globaltest 函数与所有版本保持一致，用于评估)
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc