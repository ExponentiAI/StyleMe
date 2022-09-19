import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_dwconv import DepthwiseConv2d

import math


def weights_init(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    except:
        pass


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


# DMI
class DMI(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.weight_a = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 1.01)
        self.weight_b = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.99)

        self.bias_a = nn.Parameter(torch.zeros(1, in_channels, 1, 1) + 0.01)
        self.bias_b = nn.Parameter(torch.zeros(1, in_channels, 1, 1) - 0.01)

    def forward(self, feat, mask):
        if feat.shape[1] > mask.shape[1]:
            channel_scale = feat.shape[1] // mask.shape[1]
            mask = mask.repeat(1, channel_scale, 1, 1)

        mask = F.interpolate(mask, size=feat.shape[2])
        feat_a = self.weight_a * feat * mask + self.bias_a
        feat_b = self.weight_b * feat * (1 - mask) + self.bias_b
        return feat_a + feat_b


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class Squeeze(nn.Module):
    def forward(self, feat):
        return feat.squeeze(-1).squeeze(-1)


class UnSqueeze(nn.Module):
    def forward(self, feat):
        return feat.unsqueeze(-1).unsqueeze(-1)


class ECAModule(nn.Module):
    def __init__(self, c, b=1, gamma=2):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = spectral_norm(nn.Conv1d(1, 1, k, 1, int(k / 2), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return x * out


class ResBlock(nn.Module):
    def __init__(self, ch, expansion=2):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(spectral_norm(nn.Conv2d(ch, ch * expansion, 1, 1, 0, bias=False)),
                                  spectral_norm(nn.BatchNorm2d(ch * expansion)), Swish(),
                                  spectral_norm(DepthwiseConv2d(ch * expansion, ch * expansion, 3, 1, 1)),
                                  spectral_norm(nn.BatchNorm2d(ch * expansion)), Swish(),
                                  spectral_norm(nn.Conv2d(ch * expansion, ch, 1, 1, 0, bias=False)),
                                  spectral_norm(nn.BatchNorm2d(ch)), Swish(),
                                  ECAModule(ch))

    def forward(self, x):
        return x + self.main(x)


def base_block(ch_in, ch_out):
    return nn.Sequential(nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False),
                         nn.BatchNorm2d(ch_out),
                         nn.LeakyReLU(0.2, inplace=True))


def down_block(ch_in, ch_out):
    return nn.Sequential(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(ch_out),
                         nn.LeakyReLU(0.1, inplace=True))


################################
#        style encode        #
################################

class StyleEncoder(nn.Module):
    def __init__(self, ch=32, nbr_cls=100):
        super().__init__()

        self.sf_256 = base_block(3, ch // 2)
        self.sf_128 = down_block(ch // 2, ch)
        self.sf_64 = down_block(ch, ch * 2)

        self.sf_32 = nn.Sequential(down_block(ch * 2, ch * 4),
                                   ResBlock(ch * 4))
        self.sf_16 = nn.Sequential(down_block(ch * 4, ch * 8),
                                   ResBlock(ch * 8))
        self.sf_8 = nn.Sequential(down_block(ch * 8, ch * 16),
                                  ResBlock(ch * 16))

        self.sfv_32 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=4),
                                    nn.Conv2d(ch * 4, ch * 2, 4, 1, 0, bias=False),
                                    Squeeze())
        self.sfv_16 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=4),
                                    nn.Conv2d(ch * 8, ch * 4, 4, 1, 0, bias=False),
                                    Squeeze())
        self.sfv_8 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=4),
                                   nn.Conv2d(ch * 16, ch * 8, 4, 1, 0, bias=False),
                                   Squeeze())

        self.ch = ch
        self.nbr_cls = nbr_cls
        self.final_cls = None

    def reset_cls(self):
        if self.final_cls is None:
            self.final_cls = nn.Sequential(nn.LeakyReLU(0.1), nn.Linear(self.ch * 8, self.nbr_cls))
        stdv = 1. / math.sqrt(self.final_cls[1].weight.size(1))
        self.final_cls[1].weight.data.uniform_(-stdv, stdv)
        if self.final_cls[1].bias is not None:
            self.final_cls[1].bias.data.uniform_(-0.1 * stdv, 0.1 * stdv)

    def get_feats(self, image):
        feat = self.sf_256(image)
        feat = self.sf_128(feat)
        feat = self.sf_64(feat)
        feat_32 = self.sf_32(feat)
        feat_16 = self.sf_16(feat_32)
        feat_8 = self.sf_8(feat_16)

        feat_32 = self.sfv_32(feat_32)
        feat_16 = self.sfv_16(feat_16)
        feat_8 = self.sfv_8(feat_8)

        return feat_32, feat_16, feat_8

    def forward(self, image):
        feat_32, feat_16, feat_8 = self.get_feats(image)
        pred_cls = self.final_cls(feat_8)

        return [feat_32, feat_16, feat_8], pred_cls
        #      [1, 64] [1, 128] [1, 256]


################################
#        content encode        #
################################

class ContentEncoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()

        self.feat_256 = base_block(1, ch // 4)
        self.feat_128 = down_block(ch // 4, ch // 2)
        self.feat_64 = down_block(ch // 2, ch)

        self.feat_32 = nn.Sequential(down_block(ch, ch * 2),
                                     ResBlock(ch * 2))
        self.feat_16 = nn.Sequential(down_block(ch * 2, ch * 4),
                                     ResBlock(ch * 4))
        self.feat_8 = nn.Sequential(down_block(ch * 4, ch * 8),
                                    ResBlock(ch * 8))

    def forward(self, image):
        feat = self.feat_256(image)
        feat = self.feat_128(feat)
        feat = self.feat_64(feat)

        feat_32 = self.feat_32(feat)
        feat_16 = self.feat_16(feat_32)
        feat_8 = self.feat_8(feat_16)

        return [feat_32, feat_16, feat_8]
        # [1, 64, 32, 32]
        # [1, 128, 16, 16]
        # [1, 256, 8, 8]


def for_decoder(ch_in, ch_out):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(ch_in, ch_out * 2, 3, 1, 1, bias=False),
        nn.InstanceNorm2d(ch_out * 2),
        GLU())


def style_decode(ch_in, ch_out):
    return nn.Sequential(nn.Linear(ch_in, ch_out), nn.ReLU(),
                         nn.Linear(ch_out, ch_out), nn.Sigmoid(),
                         UnSqueeze())


################################
#            decode            #
################################

class Decoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()

        self.base_feat = nn.Parameter(torch.randn(1, ch * 8, 8, 8).normal_(0, 1), requires_grad=True)

        self.dmi_8 = DMI(ch * 8)
        self.dmi_16 = DMI(ch * 4)

        self.feat_8_1 = nn.Sequential(ResBlock(ch * 16), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(ch * 16, ch * 8, 3, 1, 1, bias=False),
                                      nn.InstanceNorm2d(ch * 8))
        self.feat_8_2 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True), ResBlock(ch * 8))

        self.feat_16 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                     for_decoder(ch * 8, ch * 4), ResBlock(ch * 4))
        self.feat_32 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                     for_decoder(ch * 8, ch * 2), ResBlock(ch * 2))

        self.feat_64 = for_decoder(ch * 4, ch)
        self.feat_128 = for_decoder(ch, ch // 2)
        self.feat_256 = for_decoder(ch // 2, ch // 4)

        self.to_rgb = nn.Sequential(nn.Conv2d(ch // 4, 3, 3, 1, 1, bias=False),
                                    nn.Tanh())

        self.style_8 = style_decode(ch * 8, ch * 8)
        self.style_64 = style_decode(ch * 8, ch)
        self.style_128 = style_decode(ch * 4, ch // 2)
        self.style_256 = style_decode(ch * 2, ch // 4)

    def forward(self, content_feats, style_vectors):
        feat_8 = self.feat_8_1(torch.cat([content_feats[2],
                                          self.base_feat.repeat(style_vectors[0].shape[0], 1, 1, 1)], dim=1))
        feat_8 = self.dmi_8(feat_8, content_feats[2])

        feat_8 = feat_8 * self.style_8(style_vectors[2])
        feat_8 = self.feat_8_2(feat_8)

        feat_16 = self.feat_16(feat_8)
        feat_16 = self.dmi_16(feat_16, content_feats[1])
        feat_16 = torch.cat([feat_16, content_feats[1]], dim=1)

        feat_32 = self.feat_32(feat_16)
        feat_32 = torch.cat([feat_32, content_feats[0]], dim=1)

        feat_64 = self.feat_64(feat_32) * self.style_64(style_vectors[2])
        feat_128 = self.feat_128(feat_64) * self.style_128(style_vectors[1])
        feat_256 = self.feat_256(feat_128) * self.style_256(style_vectors[0])

        return self.to_rgb(feat_256)


################################
#           AE Module          #
################################

class AE(nn.Module):
    def __init__(self, ch, nbr_cls=100):
        super().__init__()

        self.style_encoder = StyleEncoder(ch, nbr_cls=nbr_cls)
        self.content_encoder = ContentEncoder(ch)
        self.decoder = Decoder(ch)

    @torch.no_grad()
    def forward(self, skt_img, style_img):
        style_feats = self.style_encoder.get_feats(F.interpolate(style_img, size=256))
        content_feats = self.content_encoder(F.interpolate(skt_img, size=256))
        gimg = self.decoder(content_feats, style_feats)
        return gimg, style_feats

    def load_state_dicts(self, path):
        ckpt = torch.load(path)
        self.style_encoder.reset_cls()
        self.style_encoder.load_state_dict(ckpt['s'])
        self.content_encoder.load_state_dict(ckpt['c'])
        self.decoder.load_state_dict(ckpt['d'])
        print('AE model load success')


def down_gan(ch_in, ch_out):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False)),
        nn.BatchNorm2d(ch_out),
        nn.LeakyReLU(0.1, inplace=True))


def up_gan(ch_in, ch_out):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False)),
        nn.BatchNorm2d(ch_out),
        nn.LeakyReLU(0.1, inplace=True))


def style_gan(ch_in, ch_out):
    return nn.Sequential(
        spectral_norm(nn.Linear(ch_in, ch_out)), nn.ReLU(),
        nn.Linear(ch_out, ch_out),
        nn.Sigmoid(), UnSqueeze())


################################
#              GAN             #
################################

class RefineGenerator(nn.Module):
    def __init__(self, ch=32, im_size=256):
        super().__init__()

        self.im_size = im_size

        self.from_noise_32 = nn.Sequential(UnSqueeze(),
                                           spectral_norm(nn.ConvTranspose2d(ch * 8, ch * 8, 4, 1, 0, bias=False)),
                                           nn.BatchNorm2d(ch * 8),
                                           nn.Sigmoid(),
                                           up_gan(ch * 8, ch * 4),
                                           up_gan(ch * 4, ch * 2),
                                           up_gan(ch * 2, ch * 1))

        self.from_style = nn.Sequential(UnSqueeze(),
                                        spectral_norm(
                                            nn.ConvTranspose2d(ch * (8 + 4 + 2), ch * 16, 4, 1, 0, bias=False)),
                                        nn.BatchNorm2d(ch * 16),
                                        GLU(),
                                        up_gan(ch * 8, ch * 4))

        self.encode_256 = nn.Sequential(spectral_norm(nn.Conv2d(3, ch, 3, 1, 1, bias=False)),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.encode_128 = nn.Sequential(ResBlock(ch),
                                        down_gan(ch, ch * 2))
        self.encode_64 = nn.Sequential(ResBlock(ch * 2),
                                       down_gan(ch * 2, ch * 4))
        self.encode_32 = nn.Sequential(ResBlock(ch * 4),
                                       down_gan(ch * 4, ch * 8))

        self.encode_16 = nn.Sequential(ResBlock(ch * 8),
                                       down_gan(ch * 8, ch * 16))

        self.decode_32 = nn.Sequential(ResBlock(ch * 16),
                                       up_gan(ch * 16, ch * 8))
        self.decode_64 = nn.Sequential(ResBlock(ch * 8 + ch),
                                       up_gan(ch * 8 + ch, ch * 4))
        self.decode_128 = nn.Sequential(ResBlock(ch * 4),
                                        up_gan(ch * 4, ch * 2))
        self.decode_256 = nn.Sequential(ResBlock(ch * 2),
                                        up_gan(ch * 2, ch))

        self.style_64 = style_gan(ch * 8, ch * 4)
        self.style_128 = style_gan(ch * 4, ch * 2)
        self.style_256 = style_gan(ch * 2, ch)

        self.to_rgb = nn.Sequential(nn.Conv2d(ch, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, image, style_vectors):
        n_32 = self.from_noise_32(torch.randn_like(style_vectors[2]))  # [8, 32, 32, 32]

        e_256 = self.encode_256(image)  # [8, 3, 256, 256]  [8, 32, 256, 256]
        e_128 = self.encode_128(e_256)  # [8, 64, 128, 128]
        e_64 = self.encode_64(e_128)  # [8, 128, 64, 64]
        e_32 = self.encode_32(e_64)  # [8, 256, 32, 32]

        e_16 = self.encode_16(e_32)  # [8, 256, 16, 16]

        d_32 = self.decode_32(e_16)  # [8, 256, 32, 32]
        d_64 = self.decode_64(torch.cat([d_32, n_32], dim=1))  # [8, 128, 64, 64]
        d_64 = self.style_64(style_vectors[2]) * d_64  # [8, 128, 64, 64]

        d_128 = self.decode_128(d_64 + e_64)  # [8, 64, 128, 128]
        d_128 = self.style_128(style_vectors[1]) * d_128  # [8, 64, 128, 128]

        d_256 = self.decode_256(d_128 + e_128)  # [8, 32, 256, 256]
        d_256 = self.style_256(style_vectors[0]) * d_256  # [8, 32, 256, 256]

        d_final = self.to_rgb(d_256)

        return d_final


class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.ch_out = ch_out
        self.down_main = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, feat):
        feat_out = self.down_main(feat)

        return feat_out


class Discriminator(nn.Module):
    def __init__(self, ch=64, nc=3, im_size=256):
        super(Discriminator, self).__init__()
        self.ch = ch
        self.im_size = im_size

        self.f_256 = nn.Sequential(spectral_norm(nn.Conv2d(nc, ch // 8, 3, 1, 1, bias=False)),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.f_128 = DownBlock(ch // 8, ch // 4)
        self.f_64 = DownBlock(ch // 4, ch // 2)
        self.f_32 = DownBlock(ch // 2, ch)
        self.f_16 = DownBlock(ch, ch * 2)
        self.f_8 = DownBlock(ch * 2, ch * 4)
        self.f = nn.Sequential(spectral_norm(nn.Conv2d(ch * 4, ch * 8, 1, 1, 0, bias=False)),
                               nn.BatchNorm2d(ch * 8),
                               nn.LeakyReLU(0.1, inplace=True))

        self.flatten = spectral_norm(nn.Conv2d(ch * 8, 1, 3, 1, 1, bias=False))

        self.apply(weights_init)

    def forward(self, x):
        feat_256 = self.f_256(x)
        feat_128 = self.f_128(feat_256)
        feat_64 = self.f_64(feat_128)
        feat_32 = self.f_32(feat_64)
        feat_16 = self.f_16(feat_32)
        feat_8 = self.f_8(feat_16)
        feat_f = self.f(feat_8)
        feat_out = self.flatten(feat_f)

        return feat_out
