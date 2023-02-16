##############################
#       style transform      #
##############################

import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from datasets import ImageFolder, transform_data
from models import AE, RefineGenerator


def make_matrix(dataloader_rgb, dataloader_skt, net_ae, net_ig, BATCH_SIZE, IM_SIZE, im_name):
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


if __name__ == "__main__":
    device = 'cuda'
    batch_size = 5
    img_size = 256
    num_workers = 2
    trans_iter = 20
    data_root_colorful = './train_data/rgb/'
    data_root_sketch = './train_data/sketch/'

    net_ae = AE(ch=32, nbr_cls=50)
    net_ae.style_encoder.reset_cls()
    net_ig = RefineGenerator()

    ckpt = torch.load('./checkpoint/GAN.pth')

    net_ae.load_state_dict(ckpt['ae'])
    net_ae.style_encoder.reset_cls()
    net_ig.load_state_dict(ckpt['ig'])

    net_ae.to(device)
    net_ig.to(device)
    net_ae.eval()
    net_ig.eval()

    dataset_rgb = ImageFolder(data_root_colorful, transform_data(img_size))
    dataloader_rgb = iter(DataLoader(dataset_rgb, batch_size, shuffle=False, num_workers=num_workers))

    dataset_skt = ImageFolder(data_root_sketch, transform_data(img_size))
    dataloader_skt = iter(DataLoader(dataset_skt, batch_size, shuffle=False, num_workers=num_workers))

    for idx in range(trans_iter):
        print(idx)
        make_matrix(dataloader_rgb, dataloader_skt, net_ae, net_ig, batch_size, img_size,
                    './trans_data/transform/%d.jpg' % idx)
