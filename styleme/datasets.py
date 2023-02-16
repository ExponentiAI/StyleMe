import os
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile

from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _rescale(img):
    return img * 2.0 - 1.0


def transform_data(im_size=256):
    trans = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        _rescale
    ])
    return trans


class TransformData(Dataset):
    def __init__(self, data_rgb, data_sketch, im_size=256, nbr_cls=100):
        super(TransformData, self).__init__()
        self.rgb_root = data_rgb
        self.skt_root = data_sketch

        self.frame = self._parse_frame()
        random.shuffle(self.frame)

        self.nbr_cls = nbr_cls
        self.set_offset = 0
        self.im_size = im_size

        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            _rescale
        ])

        self.transform_rd = transforms.Compose([
            transforms.Resize((int(im_size * 1.3), int(im_size * 1.3))),
            transforms.RandomCrop((int(im_size), int(im_size))),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            _rescale
        ])

        self.transform_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.8),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            _rescale
        ])

        self.transform_erase = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            _rescale,
            transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1),
            transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1),
            transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), value=1)])

        self.transform_bold = transforms.Compose([
            transforms.Resize((int(im_size * 1.1), int(im_size * 1.1))),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            _rescale
        ])

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.rgb_root)
        img_names.sort()
        for i in range(len(img_names)):
            img_name = img_names[i].zfill(len(str(len(img_names))))
            rgb_path = os.path.join(self.rgb_root, img_name)
            skt_path = os.path.join(self.skt_root, img_name)
            if os.path.exists(rgb_path) and os.path.exists(skt_path):
                frame.append((rgb_path, skt_path))

        return frame

    def __len__(self):
        return self.nbr_cls

    def _next_set(self):
        self.set_offset += self.nbr_cls
        if self.set_offset > (len(self.frame) - self.nbr_cls):
            random.shuffle(self.frame)
            self.set_offset = 0

    def __getitem__(self, idx):
        file, skt_path = self.frame[idx + self.set_offset]
        rgb = Image.open(file).convert('RGB')
        skt = Image.open(skt_path).convert('L')

        img_normal = self.transform(rgb)
        img_rd = self.transform_rd(rgb)
        img_flip = self.transform_flip(rgb)

        skt_normal = self.transform(skt)
        skt_erase = self.transform_erase(skt)
        bold_factor = 3
        skt_bold = skt.filter(ImageFilter.MinFilter(size=bold_factor))
        skt_bold = self.transform_bold(skt_bold)

        return img_normal, img_rd, img_flip, skt_normal, skt_erase, skt_bold, idx


def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class PairedDataset(Dataset):
    def __init__(self, data_root_1, data_root_2, im_size=256):
        super(PairedDataset, self).__init__()
        self.root_a = data_root_1
        self.root_b = data_root_2

        self.frame = self._parse_frame()
        self.transform = transform_data(im_size)

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root_a)
        img_names.sort()
        for i in range(len(img_names)):
            img_name = '%s.jpg' % str(i).zfill(len(str(len(img_names))))
            image_a_path = os.path.join(self.root_a, img_names[i])
            if ('.jpg' in image_a_path) or ('.png' in image_a_path):
                image_b_path = os.path.join(self.root_b, img_name)
                if os.path.exists(image_b_path):
                    frame.append((image_a_path, image_b_path))

        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file_a, file_b = self.frame[idx]
        img_a = Image.open(file_a).convert('RGB')
        img_b = Image.open(file_b).convert('L')

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return (img_a, img_b)


class ImageFolder(Dataset):
    def __init__(self, data_root, transform=transform_data(256)):
        super(ImageFolder, self).__init__()
        self.root = data_root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if ('.jpg' in image_path) or ('.png' in image_path):
                frame.append(image_path)

        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img
