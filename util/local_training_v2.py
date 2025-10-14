import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from sklearn.cluster import KMeans
from .util_v2 import get_output 

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs

    def update_weights(self, net, seed, w_g, epoch, mu=1):
        net_glob = w_g
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                if self.args.beta > 0:
                    w_diff = torch.tensor(0.).to(self.args.device)
                    for w, w_t in zip(net_glob.parameters(), net.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.beta * mu * torch.sqrt(w_diff)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # === v222: Calculate intra-class sub-prototypes ===
    def calculate_sub_prototypes(self, net):
        net.eval()
        local_sub_prototypes = {}
        loader = DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=100, shuffle=False)
        
        latent_representations, _ = get_output(loader, net.to(self.args.device), self.args, latent=True, criterion=None)
        labels = np.array(self.dataset.targets)[list(self.idxs)]

        for label in np.unique(labels):
            class_mask = (labels == label)
            class_reps = latent_representations[class_mask]
            
            n_samples_in_class = class_reps.shape[0]
            if n_samples_in_class == 0:
                continue
            
            # Ensure the number of clusters is not greater than the number of samples
            current_k = min(n_samples_in_class, self.args.k_clusters)
            
            if current_k > 0:
                kmeans = KMeans(n_clusters=current_k, random_state=self.args.seed, n_init=10).fit(class_reps)
                local_sub_prototypes[label] = kmeans.cluster_centers_
        
        return local_sub_prototypes
    
def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc