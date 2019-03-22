# coding: utf-8

import torch
import torchvision
import torchvision.transforms.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import time
import os
import PIL.Image as Image


FACESCRUB_FOLDER_TRAIN = "/path/to/your/train/data/"
FACESCRUB_FOLDER_TEST = "/path/to/your/test/data/"


root_dir_test = ""
root_dir_train = ""

dataset = "facescrub"

if dataset == "facescrub":
    root_dir_train = FACESCRUB_FOLDER_TRAIN
    root_dir_test = FACESCRUB_FOLDER_TEST
    C = 530
    train_N = 67177
    test_N = 2650


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x = np.clip(x, -1, 1)
    return x


def get_data(train = False):
    x = []
    y = []
    dir = ""
    if train:
        dir = root_dir_train
    else:
        dir = root_dir_test
    for file in os.listdir(dir):
        if file.startswith("."):
            continue
        with open(dir+file, "rb") as f:
            data, label = pickle.load(f, encoding='bytes')
            x.extend(data)
            y.extend(label)

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=int)
    x = deprocess_image(x)
    return x, y


transform = transforms.Compose([
    transforms.ToTensor()
])


class theDataset(Dataset):
    def __init__(self, transform, train ):
        super(Dataset, self).__init__()
        self.transform = transform
        self.train = train
        if self.train:
            self.train_x, self.train_y = get_data(True)
            self.train_x = self.train_x.reshape((len( self.train_x), 3, 32, 32))
            self.train_x = self.train_x.transpose((0, 2, 3, 1))
            self.train_y = self.train_y.reshape(len(self.train_y), 1)
        else:
            self.test_x, self.test_y = get_data(False)
            self.test_x = self.test_x.reshape((len(self.test_x), 3, 32, 32))
            self.test_x = self.test_x.transpose((0, 2, 3, 1))
            self.test_y = self.test_y.reshape(len(self.test_y), 1)

    def __getitem__(self, index):
        if self.train:
            imgs, labels = self.train_x[index], self.train_y[index]
        else:
            imgs, labels = self.test_x[index], self.test_y[index]

        if self.transform is not None:
            imgs = self.transform(imgs)
        imgs = imgs.type(torch.FloatTensor)
        labels = torch.from_numpy(labels)
        return imgs, labels

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)


def get_loader():
    train_set = theDataset(transform, train=True)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    test_set = theDataset(transform, train=False)
    test_loader = DataLoader(test_set, batch_size=256)
    print(len(train_set))
    print(len(test_set))
    return train_loader, test_loader


def compute_result(dataloader, net, device):
    bs = []
    label = []
    for i, (imgs, cls) in enumerate(dataloader):
        imgs = imgs.to(device)
        hash_values, _1, _2, _3 = net(imgs)
        bs.append(hash_values.data)
        label.append(cls.to(device))
    return torch.sign(torch.cat(bs)), torch.cat(label)


def timing(f):
    """print time used for function f"""
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f"[time: {time.time()-time_start}]")
        return ret
    return wrapper


@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device):
    AP = []
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        N = torch.sum(correct)
        Ns = torch.arange(1, N+1).float().to(device)
        index = (correct.nonzero() + 1)[:, 0:1].squeeze(dim=1).float()
        AP.append(torch.mean(Ns / index))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP


def encoding_onehot(target, nclasses, device):
    target_onehot = torch.Tensor(target.size(0), nclasses).to(device)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


if __name__ == '__main__':
    train_loader, test_loader = get_loader()
