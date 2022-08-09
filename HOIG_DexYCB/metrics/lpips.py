import torch
import torch.nn as nn
import argparse
import os
from torchvision import models
from data.default_dataset import get_eval_loader

def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load('metrics/lpips_weights.ckpt')
        else:
            state_dict = torch.load('metrics/lpips_weights.ckpt',
                                    map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(gen_images, gt_images):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS().eval().to(device)
    num_frames = gen_images.shape[0]
    lpips_values = []
    for i in range(num_frames):
        lpips_values.append(lpips(gen_images[i:i+1], gt_images[i:i+1]))
    lpips_values = torch.cat(lpips_values, dim=0)

    return lpips_values


@torch.no_grad()
def calculate_lpips_given_paths(paths, img_size=256, batch_size=50):
    print('Calculating LPIPS given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS().eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size, shuffle=False, drop_last=False) for path in paths]
    loaders_iter = [iter(loader) for loader in loaders]
    num_clips = (len(os.listdir(paths[0])) - 1) // batch_size + 1
    lpips_values = []

    for i in range(num_clips):
        img0 = loaders_iter[0].next().cuda()
        img1 = loaders_iter[1].next().cuda()
        lpips_values.append(lpips(img0, img1))
    lpips_values = torch.stack(lpips_values, dim=0)

    lpips_mean = torch.mean(lpips_values)

    return lpips_mean


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root_path = 'results'
    path1 = os.path.join(root_path, 'imitators')
    path2 = os.path.join(root_path, 'gt')

    lpips = calculate_lpips_given_paths([path1, path2])
    print(lpips)