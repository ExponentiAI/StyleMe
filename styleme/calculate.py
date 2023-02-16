######################################
#      calculate FID and LPIPS       #
######################################

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from datasets import PairedDataset, InfiniteSamplerWrapper
from utils import AverageMeter


def calculate_Lpips(data_root_colorful, data_root_sketch, model):
    import lpips
    from models import AE
    from models import RefineGenerator as Generator

    CHANNEL = 32
    NBR_CLS = 50
    IM_SIZE = 256
    BATCH_SIZE = 6
    DATALOADER_WORKERS = 2

    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    # load dataset
    dataset = PairedDataset(data_root_colorful, data_root_sketch, im_size=IM_SIZE)
    print('the dataset contains %d images.' % len(dataset))

    dataloader = iter(DataLoader(dataset, BATCH_SIZE, sampler=InfiniteSamplerWrapper(dataset),
                                 num_workers=DATALOADER_WORKERS, pin_memory=True))

    # load ae model
    net_ae = AE(ch=CHANNEL, nbr_cls=NBR_CLS)
    net_ae.style_encoder.reset_cls()
    net_ig = Generator(ch=CHANNEL, im_size=IM_SIZE)

    PRETRAINED_PATH = './checkpoint/GAN.pth'.format(str(model))
    print('Pre-trained path : ', PRETRAINED_PATH)
    ckpt = torch.load(PRETRAINED_PATH)

    net_ae.load_state_dict(ckpt['ae'])
    net_ig.load_state_dict(ckpt['ig'])

    net_ae.cuda()
    net_ig.cuda()
    net_ae.eval()
    net_ig.eval()

    # lpips
    get_lpips = AverageMeter()
    lpips_list = []

    # Network
    for iter_data in tqdm(range(1000)):
        rgb_img, skt_img = next(dataloader)

        rgb_img = rgb_img.cuda()
        skt_img = skt_img.cuda()

        gimg_ae, style_feats = net_ae(skt_img, rgb_img)
        g_image = net_ig(gimg_ae, style_feats)

        loss_mse = 10 * percept(F.adaptive_avg_pool2d(g_image, output_size=256),
                                F.adaptive_avg_pool2d(rgb_img, output_size=256)).sum()
        get_lpips.update(loss_mse.item() / BATCH_SIZE, BATCH_SIZE)

        lpips_list.append(get_lpips.avg)

        if (iter_data + 1) % 100 == 0:
            # print('avg : ', get_lpips.avg)
            print('LPIPS : ', sum(lpips_list) / len(lpips_list))

    print('LPIPS : ', sum(lpips_list) / len(lpips_list))


def calculate_fid(data_root_colorful, data_root_sketch, model):
    from benchmark import calc_fid, extract_feature_from_generator_fn, load_patched_inception_v3, real_image_loader, \
        image_generator, image_generator_perm
    from models import AE
    from models import RefineGenerator as Generator
    import numpy as np

    CHANNEL = 32
    NBR_CLS = 50
    IM_SIZE = 256
    BATCH_SIZE = 8
    DATALOADER_WORKERS = 2
    fid_batch_images = 119
    fid_iters = 100
    inception = load_patched_inception_v3().cuda()
    inception.eval()

    fid = []
    fid_perm = []

    # load dataset
    dataset = PairedDataset(data_root_colorful, data_root_sketch, im_size=IM_SIZE)
    print('the dataset contains %d images.' % len(dataset))

    dataloader = iter(DataLoader(dataset, BATCH_SIZE, sampler=InfiniteSamplerWrapper(dataset),
                                 num_workers=DATALOADER_WORKERS, pin_memory=True))

    # load ae model
    net_ae = AE(ch=CHANNEL, nbr_cls=NBR_CLS)
    net_ae.style_encoder.reset_cls()
    net_ig = Generator(ch=CHANNEL, im_size=IM_SIZE)

    PRETRAINED_PATH = './checkpoint/GAN.pth'.format(str(model))
    print('Pre-trained path : ', PRETRAINED_PATH)
    ckpt = torch.load(PRETRAINED_PATH)

    net_ae.load_state_dict(ckpt['ae'])
    net_ig.load_state_dict(ckpt['ig'])

    net_ae.cuda()
    net_ig.cuda()
    net_ae.eval()
    net_ig.eval()

    print("calculating FID ...")

    real_features = extract_feature_from_generator_fn(
        real_image_loader(dataloader, n_batches=fid_batch_images), inception)
    real_mean = np.mean(real_features, 0)
    real_cov = np.cov(real_features, rowvar=False)
    real_features = {'feats': real_features, 'mean': real_mean, 'cov': real_cov}

    for iter_fid in range(fid_iters):
        sample_features = extract_feature_from_generator_fn(
            image_generator(dataset, net_ae, net_ig, n_batches=fid_batch_images),
            inception, total=fid_batch_images // BATCH_SIZE - 1)
        cur_fid = calc_fid(sample_features, real_mean=real_features['mean'], real_cov=real_features['cov'])

        sample_features_perm = extract_feature_from_generator_fn(
            image_generator_perm(dataset, net_ae, net_ig, n_batches=fid_batch_images),
            inception, total=fid_batch_images // BATCH_SIZE - 1)
        cur_fid_perm = calc_fid(sample_features_perm, real_mean=real_features['mean'],
                                real_cov=real_features['cov'])

        print('FID[{}]: '.format(iter_fid), [cur_fid, cur_fid_perm])
        fid.append(cur_fid)
        fid_perm.append(cur_fid_perm)

    print('FID: ', sum(fid) / len(fid))
    print('FID perm: ', sum(fid_perm) / len(fid_perm))


if __name__ == "__main__":
    model = 'styleme'
    data_root_colorful = './train_data/comparison/rgb/'
    data_root_sketch = './train_data/comparison/sketch/'
    # data_root_colorful = './train_data/comparison/rgb/'
    # data_root_sketch = './train_data/comparison/sketch_styleme/'
    # data_root_sketch = './train_data/comparison/sketch_cam/'
    # data_root_sketch = './train_data/comparison/sketch_adalin/'
    # data_root_sketch = './train_data/comparison/sketch_wo_camada/'

    calculate_Lpips(data_root_colorful, data_root_sketch, model)
    # calculate_fid(data_root_colorful, data_root_sketch, model)

    #   styleme     |   0.13515148047968645    |    16.034930465842525
    #  styleme_wo   |   0.4334833870760152     |    32.5567679015783
    #     cam       |   0.1373054370310368     |    17.165196809300138
    #    adalin     |   0.31896749291615123    |    28.387120218137913
    #    camada     |   0.36015568705948886    |    29.75984833745646
