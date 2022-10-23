import functools
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import cat, sigmoid
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from torch.jit import ScriptModule, script_method, trace

#####################################################################
#####   functions
#####################################################################

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# def adain(content_feat, style_feat):
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    # size = content_feat.size()
    # style_mean, style_std = calc_mean_std(style_feat)
    # content_mean, content_std = calc_mean_std(content_feat)
    #
    # normalized_feat = (content_feat - content_mean.expand(
    #     size)) / content_std.expand(size)
    # return normalized_feat * style_std.expand(size) + style_mean.expand(size)
def AdaLIN(content_feat,style_feat):

        assert (content_feat.size()[:2]==style_feat.size()[:2])

        rho=Parameter(torch.Tensor(4,256,32,32,))       #维度修改了，原来是 rho=Parameter(torch.Tensor(1,512,1,1,))
        rho=rho.data.fill_(0.9)

        size=content_feat.size()
        style_mean,style_std=calc_mean_std(style_feat)
        content_mean,content_std=calc_mean_std(content_feat)
        out_style=(style_feat-style_mean.expand(size))/style_std.expand(size)
        out_content=(content_feat-content_mean.expand(size))/content_std.expand(size)
        out=rho.expand(size)*out_style+(1-rho.expand(size))*out_content
        return out

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    normalized_features = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return normalized_features  #torch.Size([4, 256, 32, 32])

def get_batched_gram_matrix(input):
    # take a batch of features: B X C X H X W
    # return gram of each image: B x C x C
    a, b, c, d = input.size()
    features = input.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(2,1)) 
    return G.div(b * c * d)
    
class Adaptive_pool(nn.Module):
    '''
    take a input tensor of size: B x C' X C'
    output a maxpooled tensor of size: B x C x H x W
    '''
    def __init__(self, channel_out, hw_out):
        super().__init__()
        self.channel_out = channel_out
        self.hw_out = hw_out
        self.pool = nn.AdaptiveAvgPool2d((channel_out, hw_out**2))
    def forward(self, input):
        if len(input.shape) == 3:
            input.unsqueeze_(1)
        return self.pool(input).view(-1, self.channel_out, self.hw_out, self.hw_out)
### new function

#####################################################################
#####   models
#####################################################################
class VGGSimple(nn.Module):
    def __init__(self):
        super(VGGSimple, self).__init__()

        self.features = self.make_layers()
        
        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, img, after_relu=True, base=4):
        # re-normalize from [-1, 1] to [0, 1] then to the range used for vgg
        feat = (((img+1)*0.5) - self.norm_mean.to(img.device)) / self.norm_std.to(img.device)
        # the layer numbers used to extract features
        cut_points = [2, 7, 14, 21, 28]
        if after_relu:
            cut_points = [c+2 for c in cut_points]
        for i in range(31):
            feat = self.features[i](feat)
            if i == cut_points[0]:
                feat_64 = F.adaptive_avg_pool2d(feat, base*16)
            if i == cut_points[1]:
                feat_32 = F.adaptive_avg_pool2d(feat, base*8)
            if i == cut_points[2]:
                feat_16 = F.adaptive_avg_pool2d(feat, base*4)
            if i == cut_points[3]:
                feat_8 = F.adaptive_avg_pool2d(feat, base*2)
            if i == cut_points[4]:
                feat_4 = F.adaptive_avg_pool2d(feat, base)
        
        return feat_64, feat_32, feat_16, feat_8, feat_4

    def make_layers(self, cfg="D", batch_norm=False):
        cfg_dic = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        cfg = cfg_dic[cfg]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)


# this model is used for pre-training
class VGG_3label(nn.Module):
    def __init__(self, nclass_artist=1117, nclass_style=55, nclass_genre=26):
        super(VGG_3label, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier_feat = self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 512))

        self.classifier_style = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(512, nclass_style))
        self.classifier_genre = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(512, nclass_genre))
        self.classifier_artist = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(512, nclass_artist))

        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    
        self.avgpool_4 = nn.AdaptiveAvgPool2d((4, 4))
        self.avgpool_8 = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool_16 = nn.AdaptiveAvgPool2d((16, 16))
    
    def get_features(self, img, after_relu=True, base=4):
        feat = (((img+1)*0.5) - self.norm_mean.to(img.device)) / self.norm_std.to(img.device)
        cut_points = [2, 7, 14, 21, 28]
        if after_relu:
            cut_points = [4, 9, 16, 23, 30]
        for i in range(31):
            feat = self.features[i](feat)
            if i == cut_points[0]:
                feat_64 = F.adaptive_avg_pool2d(feat, base*16)
            if i == cut_points[1]:
                feat_32 = F.adaptive_avg_pool2d(feat, base*8)
            if i == cut_points[2]:
                feat_16 = F.adaptive_avg_pool2d(feat, base*4)
            if i == cut_points[3]:
                feat_8 = F.adaptive_avg_pool2d(feat, base*2)
            if i == cut_points[4]:
                feat_4 = F.adaptive_avg_pool2d(feat, base)
        #feat_code = self.classifier_feat(self.avgpool(feat).view(img.size(0), -1))
        return feat_64, feat_32, feat_16, feat_8, feat_4#, feat_code


    def load_pretrain_weights(self):
        pretrained_vgg16 = vgg.vgg16(pretrained=True)
        self.features.load_state_dict(pretrained_vgg16.features.state_dict())
        self.classifier_feat[0] = pretrained_vgg16.classifier[0] 
        self.classifier_feat[3] = pretrained_vgg16.classifier[3] 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg="D", batch_norm=False):
        cfg_dic = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        cfg = cfg_dic[cfg]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, img):
        feature = self.classifier_feat( self.avgpool(self.features(img)).view(img.size(0), -1) )
        pred_style = self.classifier_style(feature)
        pred_genre = self.classifier_genre(feature)
        pred_artist = self.classifier_artist(feature)
        return pred_style, pred_genre, pred_artist


class UnFlatten(nn.Module):
    def __init__(self, block_size):
        super(UnFlatten, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.block_size, self.block_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

#batchNorm2d-->InstanceNorm2d
class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.InstanceNorm2d):
        super().__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 0, bias=True)),
            norm_layer(out_channel), 
            nn.LeakyReLU(0.01), 
            )

    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        return self.main(y)

#batchNorm2d-->InstanceNorm2d
class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.InstanceNorm2d, down=True):
        super().__init__()

        m = [   spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)),
                norm_layer(out_channel), 
                nn.LeakyReLU(0.1) ]
        if down:
            m.append(nn.AvgPool2d(2, 2))
        self.main = nn.Sequential(*m)

    def forward(self, x):
        return self.main(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self, infc=512, nfc=64, nc_out=3):
        super(Generator, self).__init__()

        self.decode_32 = UpConvBlock(infc, nfc*4)	#32
        self.decode_64 = UpConvBlock(nfc*4, nfc*4)    #64
        self.decode_128 = UpConvBlock(nfc*4, nfc*2)    #128
        self.gap_fc=nn.Linear(512,1,bias=False)
        self.gmp_fc=nn.Linear(512,1,bias=False)
        self.gamma = nn.Linear(512, 256, bias=False) #(256,256)
        self.beta = nn.Linear(512, 256, bias=False) #
        self.conv1x1 = nn.Conv2d(512, 256, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.final = nn.Sequential(
            spectral_norm( nn.Conv2d(nfc*2, nc_out, 3, 1, 1, bias=True) ),
            nn.Tanh())
        self.netG_A2B = Generator_UGATIT(image_size=256)
    def forward(self, input):

        decode_32 = self.decode_32(input)      # input torch.Size([8, 256, 32, 32])
        decode_64 = self.decode_64(decode_32)
        decode_128 = self.decode_128(decode_64)

        output = self.final(decode_128)  #output  torch.Size([8, 3, 256, 256])
        output=self.netG_A2B(output)[0]         #此处解码后，再经过Generator_UGATIT 的处理后再输出
        return output

class Generator_UGATIT(nn.Module):
    def __init__(self, image_size=256):
        super(Generator_UGATIT, self).__init__()
        down_layer = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Down-Sampling
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2, 0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3, 2, 0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Down-Sampling Bottleneck
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
        ]

        # Class Activation Map
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(512, 256, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Gamma, Beta block
        fc = [
            nn.Linear(image_size * image_size * 16, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        ]

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        # Up-Sampling Bottleneck
        for i in range(4):
            setattr(self, "ResNetAdaILNBlock_" + str(i + 1), ResNetAdaILNBlock(256))

        up_layer = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0, bias=False),
            ILN(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0, bias=False),
            ILN(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0, bias=False),
            nn.Tanh()
        ]

        self.down_layer = nn.Sequential(*down_layer)
        self.fc = nn.Sequential(*fc)
        self.up_layer = nn.Sequential(*up_layer)

    def forward(self, inputs):
        x = self.down_layer(inputs)
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        x_ = self.fc(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(4):
            x = getattr(self, "ResNetAdaILNBlock_" + str(i + 1))(x, gamma, beta)
        out = self.up_layer(x)

        return out, cam_logit




class ResNetAdaILNBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 0, bias=False)
        self.norm1 = AdaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 0, bias=False)
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + self.beta.expand(x.shape[0], -1, -1, -1)

        return out

class AdaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out







class Discriminator(nn.Module):
    def __init__(self, nfc=512, norm_layer=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(1, 4, bias=False))       #这里维度修改了原来是64 * 8, 1
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(1, 4, bias=False))
        self.conv1x1 = nn.Conv2d(2, 4, 3, 3, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(4, 4, 1, 1, 0, bias=False))

        self.main = nn.Sequential(
            DownConvBlock(nfc, nfc // 2, norm_layer=norm_layer, down=False),
            DownConvBlock(nfc // 2, nfc // 4, norm_layer=norm_layer),  # 4x4
            spectral_norm(nn.Conv2d(nfc // 4, 1, 4, 2, 0))
        )
	
    def forward(self, input):
        x = self.main(input)
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)     #x   torch.Size([4, 1, 3, 3])
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        # x = self.pad(x)
        out = self.conv(x)

        return out.view(-1)

class Discriminator_UGATIT(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator_UGATIT, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)   #input   torch.Size([1, 3, 256, 256])

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)   #x torch.Size([1, 2048, 7, 7])
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)   #out.shape  torch.Size([1, 1, 6, 6])

        return out, cam_logit, heatmap

