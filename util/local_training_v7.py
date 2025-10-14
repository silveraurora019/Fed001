# util/local_training_v7.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from sklearn.cluster import KMeans
# 导入 v7 的工具函数，它将包含 v5 的所有函数
from .util_v5 import get_output, adaptive_mcpcl_loss

# 辅助函数：Mixup 损失
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.idxs = idxs
        
        # 创新点 v7:
        # 如果使用分布感知 mixup，我们不使用标准 DataLoader，
        # 而是分析本地数据分布，以便在 update_weights 中动态采样。
        if args.use_dist_aware_mixup:
            print(f"  [Client] Activating Distribution-Aware Mixup.")
            self.dataset_split = DatasetSplit(dataset, idxs)
            self.class_indices = {c: [] for c in range(self.args.num_classes)}
            
            # 遍历本地数据集，记录每个类别的样本索引
            for i in range(len(self.dataset_split)):
                _, label = self.dataset_split[i]
                self.class_indices[label].append(i)
                
            # 记录本地拥有的类别
            self.available_classes = [c for c, indices in self.class_indices.items() if len(indices) > 0]
            
            if not self.available_classes:
                print(f"  [Client] Warning: Client has no data.")
                self.steps_per_epoch = 0
            else:
                # 计算每个 epoch 应该有多少个 step
                self.steps_per_epoch = len(self.dataset_split) // self.args.local_bs
                
        else:
            # 标准 v5 逻辑
            self.dataset_split = DatasetSplit(dataset, idxs)
            self.ldr_train = DataLoader(self.dataset_split, batch_size=self.args.local_bs, shuffle=True)

    # === NEW v7: 分布感知 mixup 批次生成器 ===
    # 灵感来源: Aorta 论文 的 Algorithm 2
    def _get_dist_aware_mixup_batch(self, batch_size):
        batch_images = []
        batch_labels_a = []
        batch_labels_b = []
        batch_lams = []

        if not self.available_classes:
            return None, None, None, None

        for _ in range(batch_size):
            # 1. 均匀采样两个类别 (c1, c2)
            # 这自动提高了少数类被选中的概率
            c1 = np.random.choice(self.available_classes)
            c2 = np.random.choice(self.available_classes)
            
            # 2. 从每个类别中均匀采样一个样本 (idx1, idx2)
            idx1 = np.random.choice(self.class_indices[c1])
            idx2 = np.random.choice(self.class_indices[c2])
            
            x1, y1 = self.dataset_split[idx1]
            x2, y2 = self.dataset_split[idx2]
            
            # 3. 生成 mixup 参数 lam
            lam = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
            
            # 4. 执行 mixup
            mixed_x = lam * x1 + (1 - lam) * x2
            
            batch_images.append(mixed_x)
            batch_labels_a.append(y1)
            batch_labels_b.append(y2)
            batch_lams.append(lam)
        
        # 转换为 Tensor
        images = torch.stack(batch_images).to(self.args.device)
        labels_a = torch.tensor(batch_labels_a).to(self.args.device).long()
        labels_b = torch.tensor(batch_labels_b).to(self.args.device).long()
        lams = torch.tensor(batch_lams).to(self.args.device).float()
        
        return images, labels_a, labels_b, lams

    # === MODIFIED v7: update_weights ===
    def update_weights(self, net, seed, w_g, epoch, mu=1, multi_client_prototypes={}, affinity_vector=None, client_id=None):
        net_glob = w_g
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        if affinity_vector is None:
            affinity_vector = torch.ones(self.args.num_users).to(self.args.device)

        epoch_loss = []
        for iter_ in range(epoch):
            batch_loss = []
            
            # === v7 逻辑分支 ===
            if self.args.use_dist_aware_mixup:
                if self.steps_per_epoch == 0:
                    continue # 跳过没有数据的客户端
                    
                for batch_idx in range(self.steps_per_epoch):
                    # 1. 获取分布感知的 mixup 批次
                    images, labels_a, labels_b, lams = self._get_dist_aware_mixup_batch(self.args.local_bs)
                    
                    net.zero_grad()
                    features = net(images, latent_output=True)
                    log_probs = net(images)
                    
                    # 2. 计算 Mixup Cross-Entropy Loss
                    loss_ce = mixup_criterion(self.loss_func, log_probs, labels_a, labels_b, lams)
                    
                    # 3. 创新点: 计算 Mixup 后的 APCL Loss (v5 组件)
                    # 我们分别计算两个标签的损失，然后用 lam 加权
                    loss_mcpcl_a = adaptive_mcpcl_loss(features, labels_a, multi_client_prototypes, affinity_vector, self.args.temp_mcpcl, self.args.device)
                    loss_mcpcl_b = adaptive_mcpcl_loss(features, labels_b, multi_client_prototypes, affinity_vector, self.args.temp_mcpcl, self.args.device)
                    loss_mcpcl = lams * loss_mcpcl_a + (1 - lams) * loss_mcpcl_b
                    # 取批次均值
                    loss_mcpcl = loss_mcpcl.mean()

                    # 4. 组合损失
                    loss = loss_ce + self.args.lambda_mcpcl * loss_mcpcl

                    if self.args.beta > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        loss += self.args.beta * mu * torch.sqrt(w_diff)

                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

            else:
                # === 标准 v5 逻辑 ===
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device).long()
                    
                    net.zero_grad()
                    features = net(images, latent_output=True)
                    log_probs = net(images)
                    
                    # 1. Standard Cross-Entropy Loss
                    loss_ce = self.loss_func(log_probs, labels)
                    
                    # 2. Adaptive Multi-Client Prototype Contrastive Loss (v5)
                    loss_mcpcl = adaptive_mcpcl_loss(features, labels, multi_client_prototypes, affinity_vector, self.args.temp_mcpcl, self.args.device)

                    # 3. Combine losses
                    loss = loss_ce + self.args.lambda_mcpcl * loss_mcpcl

                    if self.args.beta > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        loss += self.args.beta * mu * torch.sqrt(w_diff)

                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
            
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        if len(epoch_loss) == 0:
            return net.state_dict(), 0 # 处理没有数据的客户端
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def calculate_sub_prototypes(self, net):
        # (与 v5 相同)
        net.eval()
        local_sub_prototypes = {}
        # 注意：这里我们使用原始的 dataset_split，而不是 mixup_dataset 来计算原型
        loader = DataLoader(self.dataset_split, batch_size=100, shuffle=False)
        
        latent_representations, _ = get_output(loader, net.to(self.args.device), self.args, latent=True, criterion=None)
        labels = np.array(self.dataset.targets)[list(self.idxs)]

        for label in np.unique(labels):
            class_mask = (labels == label)
            class_reps = latent_representations[class_mask]
            
            n_samples_in_class = class_reps.shape[0]
            if n_samples_in_class == 0:
                continue
            
            current_k = min(n_samples_in_class, self.args.k_clusters)
            
            if current_k > 0:
                kmeans = KMeans(n_clusters=current_k, random_state=self.args.seed, n_init=10).fit(class_reps)
                local_sub_prototypes[label] = kmeans.cluster_centers_
        
        return local_sub_prototypes

def globaltest(net, test_dataset, args):
    # (与 v5 相同)
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images, latent_output=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc