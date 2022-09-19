import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import inception_v3, Inception3
from torchvision.utils import save_image
from torchvision import utils as vutils
from torch.utils.data import DataLoader

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import numpy as np
from scipy import linalg
from tqdm import tqdm
import pickle
import os
from utils import true_randperm
from datasets import InfiniteSamplerWrapper

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008,
                                    aux_logits=False,
                                    pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


@torch.no_grad()
def extract_feature_from_generator_fn(generator_fn, inception, device='cuda', total=1000):
    features = []

    for batch in tqdm(generator_fn, total=total):
        try:
            feat = inception(batch)[0].view(batch.shape[0], -1)
            features.append(feat.to('cpu'))
        except:
            break
    features = torch.cat(features, 0).detach()
    return features.numpy()


def calc_fid(sample_features, real_features=None, real_mean=None, real_cov=None, eps=1e-6):
    sample_mean = np.mean(sample_features, 0)
    sample_cov = np.cov(sample_features, rowvar=False)

    if real_features is not None:
        real_mean = np.mean(real_features, 0)
        real_cov = np.cov(real_features, rowvar=False)

    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def real_image_loader(dataloader, n_batches=10):
    counter = 0
    while counter < n_batches:
        counter += 1
        rgb_img = next(dataloader)[0]
        if counter == 1:
            vutils.save_image(0.5 * (rgb_img + 1), './checkpoint/tmp_real.jpg')
        yield rgb_img.cuda()


@torch.no_grad()
def image_generator(dataset, net_ae, net_ig, BATCH_SIZE=8, n_batches=500):
    counter = 0
    dataloader = iter(
        DataLoader(dataset, BATCH_SIZE, sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=False))
    n_batches = min(n_batches, len(dataset) // BATCH_SIZE - 1)
    while counter < n_batches:
        counter += 1
        rgb_img, skt_img = next(dataloader)
        rgb_img = F.interpolate(rgb_img, size=256).cuda()
        skt_img = F.interpolate(skt_img, size=256).cuda()

        gimg_ae, style_feat = net_ae(skt_img, rgb_img)
        g_image = net_ig(gimg_ae, style_feat)
        if counter == 1:
            vutils.save_image(0.5 * (g_image + 1), './checkpoint/tmp.jpg')
        yield g_image


@torch.no_grad()
def image_generator_perm(dataset, net_ae, net_ig, BATCH_SIZE=8, n_batches=500):
    counter = 0
    dataloader = iter(DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False))
    n_batches = min(n_batches, len(dataset) // BATCH_SIZE - 1)
    while counter < n_batches:
        counter += 1
        rgb_img, skt_img = next(dataloader)
        rgb_img = F.interpolate(rgb_img, size=256).cuda()
        skt_img = F.interpolate(skt_img, size=256).cuda()

        perm = true_randperm(rgb_img.shape[0], device=rgb_img.device)

        gimg_ae, style_feat = net_ae(skt_img, rgb_img[perm])
        g_image = net_ig(gimg_ae, style_feat)
        if counter == 1:
            vutils.save_image(0.5 * (g_image + 1), './checkpoint/tmp.jpg')
        yield g_image


if __name__ == "__main__":
    from utils import PairedMultiDataset, InfiniteSamplerWrapper, make_folders, AverageMeter
    from torch.utils.data import DataLoader
    from torchvision import utils as vutils

    IM_SIZE = 1024
    BATCH_SIZE = 8
    DATALOADER_WORKERS = 8
    NBR_CLS = 2000
    TRIAL_NAME = 'trial_vae_512_1'
    SAVE_FOLDER = './'

    data_root_colorful = '../images/celebA/CelebA_512_test/img'
    data_root_sketch_1 = './sketch_simplification/vggadin_iter_700_test'
    data_root_sketch_2 = './sketch_simplification/vggadin_iter_1900_test'
    data_root_sketch_3 = './sketch_simplification/vggadin_iter_2300_test'

    dataset = PairedMultiDataset(data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3,
                                 im_size=IM_SIZE, rand_crop=False)
    dataloader = iter(DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS, pin_memory=True))

    from models import StyleEncoder, ContentEncoder, Decoder
    import pickle
    from models import AE, RefineGenerator
    from utils import load_params

    net_ig = RefineGenerator().cuda()
    net_ig = nn.DataParallel(net_ig)

    ckpt = './train_results/trial_refine_ae_as_gan_1024_2/models/4.pth'
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        # net_ig.load_state_dict(ckpt['ig'])
        # net_id.load_state_dict(ckpt['id'])
        net_ig_ema = ckpt['ig_ema']
        load_params(net_ig, net_ig_ema)
    net_ig = net_ig.module
    # net_ig.eval()

    net_ae = AE()
    net_ae.load_state_dicts('./train_results/trial_vae_512_1/models/176000.pth')
    net_ae.cuda()
    net_ae.eval()

    inception = load_patched_inception_v3().cuda()
    inception.eval()

    '''
    real_features = extract_feature_from_generator_fn( 
        real_image_loader(dataloader, n_batches=1000), inception )
    real_mean = np.mean(real_features, 0)
    real_cov = np.cov(real_features, rowvar=False)
    '''
    # pickle.dump({'feats': real_features, 'mean': real_mean, 'cov': real_cov}, open('celeba_fid_feats.npy','wb') )

    real_features = pickle.load(open('celeba_fid_feats.npy', 'rb'))
    real_mean = real_features['mean']
    real_cov = real_features['cov']
    # sample_features = extract_feature_from_generator_fn( real_image_loader(dataloader, n_batches=100), inception )
    for it in range(1):
        itx = it * 8000
        '''
        ckpt = torch.load('./train_results/%s/models/%d.pth'%(TRIAL_NAME, itx))

        style_encoder.load_state_dict(ckpt['e'])
        content_encoder.load_state_dict(ckpt['c'])
        decoder.load_state_dict(ckpt['d'])
        
        dataloader = iter(DataLoader(dataset, BATCH_SIZE, sampler=InfiniteSamplerWrapper(dataset), num_workers=DATALOADER_WORKERS, pin_memory=True))
        '''

        sample_features = extract_feature_from_generator_fn(
            image_generator(dataset, net_ae, net_ig, n_batches=1800), inception,
            total=1800)

        # fid = calc_fid(sample_features, real_mean=real_features['mean'], real_cov=real_features['cov'])
        fid = calc_fid(sample_features, real_mean=real_mean, real_cov=real_cov)

        print(it, fid)
