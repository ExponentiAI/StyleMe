import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
from tqdm import tqdm
import pickle
import numpy as np

from datasets import PairedDataset, InfiniteSamplerWrapper
from utils import copy_G_params, load_params, AverageMeter, make_folders, d_hinge_loss, g_hinge_loss
from models import AE, Discriminator


def make_matrix(dataset_rgb, dataset_skt, net_ae, net_ig, BATCH_SIZE, IM_SIZE, im_name):
    dataloader_rgb = iter(DataLoader(dataset_rgb, BATCH_SIZE, shuffle=True))
    dataloader_skt = iter(DataLoader(dataset_skt, BATCH_SIZE, shuffle=True))

    rgb_img = next(dataloader_rgb)
    skt_img = next(dataloader_skt)

    skt_img = skt_img.mean(dim=1, keepdim=True)

    image_matrix = [torch.ones(1, 3, IM_SIZE, IM_SIZE)]
    image_matrix.append(rgb_img.clone())
    with torch.no_grad():
        rgb_img = rgb_img.cuda()
        for skt in skt_img:
            input_skts = skt.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).cuda()

            gimg_ae, style_feats = net_ae(input_skts, rgb_img)
            image_matrix.append(skt.unsqueeze(0).repeat(1, 3, 1, 1).clone())
            image_matrix.append(gimg_ae.cpu())

            g_images = net_ig(gimg_ae, style_feats).cpu()
            image_matrix.append(skt.unsqueeze(0).repeat(1, 3, 1, 1).clone().fill_(1))
            image_matrix.append(torch.nn.functional.interpolate(g_images, IM_SIZE))

    image_matrix = torch.cat(image_matrix)
    vutils.save_image(0.5 * (image_matrix + 1), im_name, nrow=BATCH_SIZE + 1)


def train():
    from benchmark import calc_fid, extract_feature_from_generator_fn, load_patched_inception_v3, real_image_loader, \
        image_generator, image_generator_perm
    import lpips

    from config import IM_SIZE_GAN, BATCH_SIZE_GAN, CHANNEL, NBR_CLS, DATALOADER_WORKERS, EPOCH_GAN, ITERATION_GAN, \
        ITERATION_AE, GAN_CKECKPOINT
    from config import SAVE_IMAGE_INTERVAL, SAVE_MODEL_INTERVAL, LOG_INTERVAL, SAVE_FOLDER, MULTI_GPU
    from config import FID_INTERVAL, FID_BATCH_NBR, PRETRAINED_AE_PATH
    from config import data_root_colorful, data_root_sketch

    real_features = None
    inception = load_patched_inception_v3().cuda()
    inception.eval()

    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    # save path
    saved_image_folder = saved_model_folder = None
    log_file_path = None
    if saved_image_folder is None:
        saved_image_folder, saved_model_folder = make_folders(SAVE_FOLDER, 'Train_GAN')
        log_file_path = saved_image_folder + '/../gan_log.txt'
        log_file = open(log_file_path, 'w')
        log_file.close()

    # load dataset
    dataset = PairedDataset(data_root_colorful, data_root_sketch, im_size=IM_SIZE_GAN)
    print('the dataset contains %d images.' % len(dataset))
    dataloader = iter(DataLoader(dataset, BATCH_SIZE_GAN, sampler=InfiniteSamplerWrapper(dataset),
                                 num_workers=DATALOADER_WORKERS, pin_memory=True))

    from datasets import ImageFolder, transform_data

    dataset_rgb = ImageFolder(data_root_colorful, transform_data(IM_SIZE_GAN))
    dataset_skt = ImageFolder(data_root_sketch, transform_data(IM_SIZE_GAN))

    # load ae model
    net_ae = AE(ch=CHANNEL, nbr_cls=NBR_CLS)

    if PRETRAINED_AE_PATH is None:
        PRETRAINED_AE_PATH = SAVE_FOLDER + 'train_results/Train_AE/' + 'models/%d.pth' % ITERATION_AE
    else:
        PRETRAINED_AE_PATH = PRETRAINED_AE_PATH

    print('Pre-trained AE path : ', PRETRAINED_AE_PATH)

    net_ae.load_state_dicts(PRETRAINED_AE_PATH)
    net_ae.cuda()
    net_ae.eval()

    from models import RefineGenerator as Generator

    # load generator & discriminator
    net_ig = Generator(ch=CHANNEL, im_size=IM_SIZE_GAN).cuda()
    net_id = Discriminator(nc=3).cuda()

    if MULTI_GPU:
        net_ae = nn.DataParallel(net_ae)
        net_ig = nn.DataParallel(net_ig)
        net_id = nn.DataParallel(net_id)

    net_ig_ema = copy_G_params(net_ig)

    opt_ig = optim.Adam(net_ig.parameters(), lr=2e-4, betas=(0.8, 0.999))
    opt_id = optim.Adam(net_id.parameters(), lr=2e-4, betas=(0.8, 0.999))

    if GAN_CKECKPOINT is not None:
        ckpt = torch.load(GAN_CKECKPOINT)
        net_ig.load_state_dict(ckpt['ig'])
        net_id.load_state_dict(ckpt['id'])
        net_ig_ema = ckpt['ig_ema']
        opt_ig.load_state_dict(ckpt['opt_ig'])
        opt_id.load_state_dict(ckpt['opt_id'])

    # loss log
    losses_g_img = AverageMeter()
    losses_d_img = AverageMeter()
    losses_mse = AverageMeter()
    losses_style = AverageMeter()
    losses_content = AverageMeter()
    losses_rec_ae = AverageMeter()

    fixed_skt = fixed_rgb = None

    fid = [[0, 0]]

    ###################
    #    train gan    #
    ###################
    for epoch in range(EPOCH_GAN):
        for iteration in tqdm(range(ITERATION_GAN)):
            rgb_img, skt_img = next(dataloader)

            rgb_img = rgb_img.cuda()
            skt_img = skt_img.cuda()

            if iteration == 0:
                fixed_skt = skt_img[:8].clone().cuda()
                fixed_rgb = rgb_img[:8].clone()

            # 1. train Discriminator
            gimg_ae, style_feats = net_ae(skt_img, rgb_img)
            g_image = net_ig(gimg_ae, style_feats)

            real = net_id(rgb_img)
            fake = net_id(g_image.detach())

            loss_d = d_hinge_loss(real, fake)

            net_id.zero_grad()
            loss_d.backward()
            opt_id.step()

            # log ae loss
            loss_rec_ae = F.mse_loss(gimg_ae, rgb_img) + F.l1_loss(gimg_ae, rgb_img)
            losses_rec_ae.update(loss_rec_ae.item(), BATCH_SIZE_GAN)

            # 2. train Generator
            pred_g = net_id(g_image)
            loss_g = g_hinge_loss(pred_g)

            loss_mse = 10 * percept(F.adaptive_avg_pool2d(g_image, output_size=256),
                                    F.adaptive_avg_pool2d(rgb_img, output_size=256)).sum()
            losses_mse.update(loss_mse.item() / BATCH_SIZE_GAN, BATCH_SIZE_GAN)

            _, g_style_feats = net_ae(skt_img, g_image)

            loss_style = 0
            for loss_idx in range(3):
                loss_style += - F.cosine_similarity(g_style_feats[loss_idx],
                                                    style_feats[loss_idx].detach()).mean() + \
                              F.cosine_similarity(g_style_feats[loss_idx],
                                                  style_feats[loss_idx][torch.randperm(BATCH_SIZE_GAN)]
                                                  .detach()).mean()
            losses_style.update(loss_style.item() / BATCH_SIZE_GAN, BATCH_SIZE_GAN)

            loss_all = loss_g + loss_mse + loss_style

            net_ig.zero_grad()
            loss_all.backward()
            opt_ig.step()

            for p, avg_p in zip(net_ig.parameters(), net_ig_ema):
                avg_p.mul_(0.999).add_(p.data, alpha=0.001)

            # 3. logging
            losses_g_img.update(pred_g.mean().item(), BATCH_SIZE_GAN)
            losses_d_img.update(real.mean().item(), BATCH_SIZE_GAN)

            if iteration % SAVE_IMAGE_INTERVAL == 0:
                with torch.no_grad():
                    backup_params_g = copy_G_params(net_ig)
                    load_params(net_ig, net_ig_ema)

                    gimg_ae, style_feats = net_ae(fixed_skt, fixed_rgb)
                    img_net_g = net_ig(gimg_ae, style_feats)

                    gimg = torch.cat([F.interpolate(fixed_rgb, IM_SIZE_GAN),
                                      F.interpolate(fixed_skt.repeat(1, 3, 1, 1), IM_SIZE_GAN),
                                      F.interpolate(gimg_ae, IM_SIZE_GAN),
                                      img_net_g])

                    vutils.save_image(gimg, f'{saved_image_folder}/{epoch}_{iteration}.jpg', normalize=True,
                                      range=(-1, 1))
                    del gimg

                    make_matrix(dataset_rgb, dataset_skt, net_ae, net_ig, 5, IM_SIZE_GAN,
                                f'{saved_image_folder}/{epoch}_{iteration}_matrix.jpg')

                    load_params(net_ig, backup_params_g)

            # 4. print log
            if iteration % LOG_INTERVAL == 0:
                log_msg = ' \nGAN_Iter: [{0}/{1}]   AE_loss: {ae_loss: .5f} \n' \
                          'Generator: {losses_g_img.avg:.4f}  Discriminator: {losses_d_img.avg:.4f} \n' \
                          'Style: {losses_style.avg:.5f}  Content: {losses_content.avg:.5f}  \n' \
                          'MSE: {losses_mse.avg:.4f}  FID: {fid:.4f} \n'.format(
                    epoch, iteration, ae_loss=losses_rec_ae.avg, losses_g_img=losses_g_img, losses_d_img=losses_d_img,
                    losses_style=losses_style, losses_content=losses_content, losses_mse=losses_mse, fid=fid[-1][0])

                print(log_msg)

                if log_file_path is not None:
                    log_file = open(log_file_path, 'a')
                    log_file.write(log_msg + '\n')
                    log_file.close()

                losses_g_img.reset()
                losses_d_img.reset()
                losses_mse.reset()
                losses_style.reset()
                losses_content.reset()
                losses_rec_ae.reset()

            # 5. save model
            if iteration % SAVE_MODEL_INTERVAL == 0 or iteration + 1 == 10000:
                print('Saving history model')
                torch.save({'ig': net_ig.state_dict(),
                            'id': net_id.state_dict(),
                            'ae': net_ae.state_dict(),
                            'ig_ema': net_ig_ema,
                            'opt_ig': opt_ig.state_dict(),
                            'opt_id': opt_id.state_dict(),
                            }, '%s/%d.pth' % (saved_model_folder, epoch))

            # 6. FID
            if iteration % FID_INTERVAL == 0 and iteration > 1:
                print("calculating FID ...")
                fid_batch_images = FID_BATCH_NBR
                if real_features is None:
                    if os.path.exists('fid_feats.npy'):
                        real_features = pickle.load(open('fid_feats.npy', 'rb'))
                    else:
                        real_features = extract_feature_from_generator_fn(
                            real_image_loader(dataloader, n_batches=fid_batch_images), inception)
                        real_mean = np.mean(real_features, 0)
                        real_cov = np.cov(real_features, rowvar=False)
                        pickle.dump({'feats': real_features, 'mean': real_mean, 'cov': real_cov},
                                    open('fid_feats.npy', 'wb'))
                        real_features = pickle.load(open('fid_feats.npy', 'rb'))

                sample_features = extract_feature_from_generator_fn(
                    image_generator(dataset, net_ae, net_ig, n_batches=fid_batch_images), inception,
                    total=fid_batch_images)
                cur_fid = calc_fid(sample_features, real_mean=real_features['mean'], real_cov=real_features['cov'])
                sample_features_perm = extract_feature_from_generator_fn(
                    image_generator_perm(dataset, net_ae, net_ig, n_batches=fid_batch_images), inception,
                    total=fid_batch_images)
                cur_fid_perm = calc_fid(sample_features_perm, real_mean=real_features['mean'],
                                        real_cov=real_features['cov'])

                fid.append([cur_fid, cur_fid_perm])
                print('\nFID: ', fid)
                if log_file_path is not None:
                    log_file = open(log_file_path, 'a')
                    log_msg = 'fid: %.5f, %.5f' % (fid[-1][0], fid[-1][1])
                    log_file.write(log_msg + '\n')
                    log_file.close()


if __name__ == "__main__":
    train()
