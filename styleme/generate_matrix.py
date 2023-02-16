import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils


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
