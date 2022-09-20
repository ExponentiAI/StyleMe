import os
import torch
from copy import deepcopy
from random import shuffle
import torch.nn.functional as F


def d_hinge_loss(real_pred, fake_pred):
    real_loss = F.relu(1 - real_pred)
    fake_loss = F.relu(1 + fake_pred)

    return real_loss.mean() + fake_loss.mean()


def g_hinge_loss(pred):
    return -pred.mean()


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def true_randperm(size, device='cuda'):
    def unmatched_randperm(size):
        l1 = [i for i in range(size)]
        l2 = []
        for j in range(size):
            deleted = False
            if j in l1:
                deleted = True
                del l1[l1.index(j)]
            shuffle(l1)
            if len(l1) == 0:
                return 0, False
            l2.append(l1[0])
            del l1[0]
            if deleted:
                l1.append(j)
        return l2, True

    flag = False
    l = torch.zeros(size).long()
    while not flag:
        l, flag = unmatched_randperm(size)
    return torch.LongTensor(l).to(device)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def make_folders(save_folder, trial_name):
    saved_model_folder = os.path.join(save_folder, 'train_results/%s/models' % trial_name)
    saved_image_folder = os.path.join(save_folder, 'train_results/%s/images' % trial_name)
    folders = [os.path.join(save_folder, 'train_results'),
               os.path.join(save_folder, 'train_results/%s' % trial_name),
               os.path.join(save_folder, 'train_results/%s/images' % trial_name),
               os.path.join(save_folder, 'train_results/%s/models' % trial_name)]
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    from shutil import copy
    try:
        for f in os.listdir('.'):
            if '.py' in f:
                copy(f, os.path.join(save_folder, 'train_results/%s' % trial_name) + '/' + f)
    except:
        pass
    return saved_image_folder, saved_model_folder
