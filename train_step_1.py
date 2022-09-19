import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import random
from tqdm import tqdm

from datasets import TransformData, InfiniteSamplerWrapper
from utils import make_folders, AverageMeter
from models import StyleEncoder, ContentEncoder, Decoder


def loss_for_style(style, style_org, batch_size):
    loss_result = 0
    for loss_idx in range(len(style)):
        loss_result += - F.cosine_similarity(style[loss_idx],
                                               style_org[loss_idx].detach()).mean() + \
                       F.cosine_similarity(style[loss_idx],
                                           style_org[loss_idx][torch.randperm(batch_size)]
                                           .detach()).mean()
    return loss_result / len(style)


def loss_for_content(loss, fl1, fl2):
    loss_result = 0
    for f_idx in range(len(fl1)):
        loss_result += loss(fl1[f_idx], fl2[f_idx].detach())
    return loss_result * 2


def train():
    from config import IM_SIZE_AE, BATCH_SIZE_AE, CHANNEL, NBR_CLS, DATALOADER_WORKERS, ITERATION_AE
    from config import SAVE_IMAGE_INTERVAL, SAVE_MODEL_INTERVAL, SAVE_FOLDER, LOG_INTERVAL
    from config import data_root_colorful, data_root_sketch

    dataset_trans = TransformData(data_root_colorful, data_root_sketch, im_size=IM_SIZE_AE, nbr_cls=NBR_CLS)
    print('Num classes:', len(dataset_trans), '  Data nums:', len(dataset_trans.frame))
    dataloader_trans = iter(DataLoader(dataset_trans, BATCH_SIZE_AE,
                                       sampler=InfiniteSamplerWrapper(dataset_trans),
                                       num_workers=DATALOADER_WORKERS, pin_memory=True))

    style_encoder = StyleEncoder(ch=CHANNEL, nbr_cls=NBR_CLS).cuda()
    content_encoder = ContentEncoder(ch=CHANNEL).cuda()
    decoder = Decoder(ch=CHANNEL).cuda()

    opt_content = optim.Adam(content_encoder.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_style = optim.Adam(style_encoder.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_decode = optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.9, 0.999))

    style_encoder.reset_cls()
    style_encoder.final_cls.cuda()

    # load model
    from config import PRETRAINED_AE_PATH
    if PRETRAINED_AE_PATH is not None:
        ckpt = torch.load(PRETRAINED_AE_PATH)

        print('Pre-trained AE path : ', PRETRAINED_AE_PATH)

        style_encoder.load_state_dict(ckpt['s'])
        content_encoder.load_state_dict(ckpt['c'])
        decoder.load_state_dict(ckpt['d'])

        opt_style.load_state_dict(ckpt['opt_s'])
        opt_content.load_state_dict(ckpt['opt_c'])
        opt_decode.load_state_dict(ckpt['opt_d'])
        print('loaded pre-trained AE')

    style_encoder.reset_cls()
    style_encoder.final_cls.cuda()
    opt_s_cls = optim.Adam(style_encoder.final_cls.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # save path
    saved_image_folder, saved_model_folder = make_folders(SAVE_FOLDER, 'Train_AE')
    log_file_path = saved_image_folder + '/../ae_log.txt'
    log_file = open(log_file_path, 'w')
    log_file.close()

    # loss log
    losses_style_feat = AverageMeter()
    losses_content_feat = AverageMeter()
    losses_cls = AverageMeter()
    losses_org = AverageMeter()
    losses_rd = AverageMeter()
    losses_flip = AverageMeter()

    import lpips
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

    for iteration in tqdm(range(ITERATION_AE)):
        if iteration % ((NBR_CLS * 100) // BATCH_SIZE_AE) == 0 and iteration > 1:
            dataset_trans._next_set()
            dataloader_trans = iter(DataLoader(dataset_trans, BATCH_SIZE_AE,
                                               sampler=InfiniteSamplerWrapper(dataset_trans),
                                               num_workers=DATALOADER_WORKERS, pin_memory=True))
            style_encoder.reset_cls()
            opt_s_cls = optim.Adam(style_encoder.final_cls.parameters(), lr=1e-4, betas=(0.9, 0.999))

            opt_style.param_groups[0]['lr'] = 1e-4
            opt_decode.param_groups[0]['lr'] = 1e-4

        # 1. training for encode & decode
        # 1.1 prepare data
        rgb_img_org, rgb_img_rd, rgb_img_flip, skt_org, skt_erased, skt_bold, img_idx = next(dataloader_trans)
        rgb_img_org = rgb_img_org.cuda()
        rgb_img_rd = rgb_img_rd.cuda()
        rgb_img_flip = rgb_img_flip.cuda()

        skt_org = F.interpolate(skt_org, size=IM_SIZE_AE).cuda()
        skt_erased = F.interpolate(skt_erased, size=IM_SIZE_AE).cuda()
        skt_bold = F.interpolate(skt_bold, size=IM_SIZE_AE).cuda()

        img_idx = img_idx.long().cuda()

        # 1.2 model grad zero
        style_encoder.zero_grad()
        content_encoder.zero_grad()
        decoder.zero_grad()

        ################
        #    encode    #
        ################
        # 1.3 for style
        style_vector_org, pred_cls_org = style_encoder(rgb_img_org)
        style_vector_rd, pred_cls_rd = style_encoder(rgb_img_rd)
        style_vector_flip, pred_cls_flip = style_encoder(rgb_img_flip)

        # 1.4 for content
        content_feats_org = content_encoder(skt_org)
        content_feats_erased = content_encoder(skt_erased)
        content_feats_bold = content_encoder(skt_bold)

        # 1.5 encode loss
        loss_style_feat = loss_for_style(style_vector_rd, style_vector_org, BATCH_SIZE_AE) + \
                          loss_for_style(style_vector_flip, style_vector_org, BATCH_SIZE_AE)

        loss_content_feat = loss_for_content(F.mse_loss, content_feats_bold, content_feats_org) + \
                            loss_for_content(F.mse_loss, content_feats_erased, content_feats_org)

        loss_cls = F.cross_entropy(pred_cls_org, img_idx) + \
                   F.cross_entropy(pred_cls_rd, img_idx) + \
                   F.cross_entropy(pred_cls_flip, img_idx)

        ################
        #    decode    #
        ################
        org = random.randint(0, 2)
        gimg_org = None
        if org == 0:
            gimg_org = decoder(content_feats_org, style_vector_org)
        elif org == 1:
            gimg_org = decoder(content_feats_erased, style_vector_org)
        elif org == 2:
            gimg_org = decoder(content_feats_bold, style_vector_org)

        rd = random.randint(0, 2)
        gimg_rd = None
        if rd == 0:
            gimg_rd = decoder(content_feats_org, style_vector_rd)
        elif rd == 1:
            gimg_rd = decoder(content_feats_erased, style_vector_rd)
        elif rd == 2:
            gimg_rd = decoder(content_feats_bold, style_vector_rd)

        flip = random.randint(0, 2)
        gimg_flip = None
        if flip == 0:
            gimg_flip = decoder(content_feats_org, style_vector_flip)
        elif flip == 1:
            gimg_flip = decoder(content_feats_erased, style_vector_flip)
        elif flip == 2:
            gimg_flip = decoder(content_feats_bold, style_vector_flip)

        # 1.6 decode loss
        loss_org = F.mse_loss(gimg_org, rgb_img_org) + \
                   percept(F.adaptive_avg_pool2d(gimg_org, output_size=256),
                           F.adaptive_avg_pool2d(rgb_img_org, output_size=256)).sum()

        loss_rd = F.mse_loss(gimg_rd, rgb_img_org) + \
                  percept(F.adaptive_avg_pool2d(gimg_rd, output_size=256),
                          F.adaptive_avg_pool2d(rgb_img_org, output_size=256)).sum()

        loss_flip = F.mse_loss(gimg_flip, rgb_img_org) + \
                    percept(F.adaptive_avg_pool2d(gimg_flip, output_size=256),
                            F.adaptive_avg_pool2d(rgb_img_org, output_size=256)).sum()

        loss_total = loss_style_feat + loss_content_feat + loss_cls + loss_org + loss_rd + loss_flip
        loss_total.backward()

        opt_style.step()
        opt_content.step()
        opt_s_cls.step()
        opt_decode.step()

        # 1.7 update log
        losses_style_feat.update(loss_style_feat.mean().item(), BATCH_SIZE_AE)
        losses_content_feat.update(loss_content_feat.mean().item(), BATCH_SIZE_AE)
        losses_cls.update(loss_cls.mean().item(), BATCH_SIZE_AE)
        losses_org.update(loss_org.item(), BATCH_SIZE_AE)
        losses_rd.update(loss_rd.item(), BATCH_SIZE_AE)
        losses_flip.update(loss_flip.item(), BATCH_SIZE_AE)

        # 1.8 print log
        if iteration % LOG_INTERVAL == 0:
            log_msg = '\nTrain Stage 1 (encode and decode): \n' \
                      'loss_encode_style: %.4f     loss_encode_content: %.4f     loss_encode_class: %.4f  \n' \
                      'loss_decode_org: %.4f       loss_decode_rd: %.4f          loss_decode_flip: %.4f' % (
                          losses_style_feat.avg, losses_content_feat.avg, losses_cls.avg,
                          losses_org.avg, losses_rd.avg, losses_flip.avg)
            print(log_msg)

            if log_file_path is not None:
                log_file = open(log_file_path, 'a')
                log_file.write(log_msg + '\n')
                log_file.close()

            losses_style_feat.reset()
            losses_content_feat.reset()
            losses_cls.reset()
            losses_org.reset()
            losses_rd.reset()
            losses_flip.reset()

        if iteration % SAVE_IMAGE_INTERVAL == 0:
            vutils.save_image(torch.cat([rgb_img_org,
                                         F.interpolate(skt_org.repeat(1, 3, 1, 1), size=IM_SIZE_AE),
                                         gimg_org]),
                              '%s/%d_org.jpg' % (saved_image_folder, iteration), normalize=True, range=(-1, 1))
            vutils.save_image(torch.cat([rgb_img_rd,
                                         F.interpolate(skt_org.repeat(1, 3, 1, 1), size=IM_SIZE_AE),
                                         gimg_rd]),
                              '%s/%d_rd.jpg' % (saved_image_folder, iteration), normalize=True, range=(-1, 1))
            vutils.save_image(torch.cat([rgb_img_flip,
                                         F.interpolate(skt_org.repeat(1, 3, 1, 1), size=IM_SIZE_AE),
                                         gimg_flip]),
                              '%s/%d_flip.jpg' % (saved_image_folder, iteration), normalize=True, range=(-1, 1))

        if iteration % SAVE_MODEL_INTERVAL == 0:
            print('Saving history model')
            torch.save({'s': style_encoder.state_dict(),
                        'c': content_encoder.state_dict(),
                        'd': decoder.state_dict(),
                        'opt_s': opt_style.state_dict(),
                        'opt_c': opt_content.state_dict(),
                        'opt_s_cls': opt_s_cls.state_dict(),
                        'opt_d': opt_decode.state_dict(),
                        }, '%s/%d.pth' % (saved_model_folder, iteration))

    torch.save({'s': style_encoder.state_dict(),
                'c': content_encoder.state_dict(),
                'd': decoder.state_dict(),
                'opt_s': opt_style.state_dict(),
                'opt_c': opt_content.state_dict(),
                'opt_s_cls': opt_s_cls.state_dict(),
                'opt_d': opt_decode.state_dict(),
                }, '%s/%d.pth' % (saved_model_folder, ITERATION_AE))


if __name__ == "__main__":
    train()
