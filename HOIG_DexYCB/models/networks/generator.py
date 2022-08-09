import torch.nn as nn
import torch.nn.functional as F
from .base_network import NetworkBase
from .extract_attn import ExtractorAttn
from .spade import SPADE
import torch


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.learned_shortcut = (dim_in != dim_out)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
            self.norm_s = nn.InstanceNorm2d(dim_out, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def forward(self, x):
        return self.shortcut(x) + self.main(x)


class SPADEResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(SPADEResidualBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (dim_in != dim_out)

        # create conv layers
        self.conv_0 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = SPADE(dim_in, dim_c)
        self.norm_1 = SPADE(dim_out, dim_c)
        if self.learned_shortcut:
            self.norm_s = SPADE(dim_in, dim_c)

        # define activation function
        self.actvn = nn.ReLU(inplace=True)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out


class SPADEBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, kernel_size=3, Downsample=True):
        super(SPADEBlock, self).__init__()
        # Attributes
        if Downsample:
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=2, padding=1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=2, padding=1, output_padding=1, bias=False)
        self.norm = SPADE(dim_out, dim_c)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x, seg):
        x = self.conv(x)
        x = self.norm(x, seg)
        x = self.actv(x)

        return x


class ResNetGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=9, k_size=4, n_down=2):
        super(ResNetGenerator, self).__init__()
        self._name = 'resnet_generator'

        layers = []
        layers.append(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(n_down):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1,
                                             output_padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x, c=None):
        if c is not None:
            # replicate spatially and concatenate domain information
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
        return self.model(x)


class ResUnetGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, k_size=4, n_down=2, s_dim=0, spade_layers=[0, 0, 0, 0], on_obj=False):
        super(ResUnetGenerator, self).__init__()
        self._name = 'resunet_generator'

        self.repeat_num = repeat_num
        self.n_down = n_down
        self.spade_layers = spade_layers
        self.on_obj = on_obj
        self.num_channel = dict()

        encoders = []

        encoders.append(nn.Sequential(
            nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        ))
        self.num_channel[0] = conv_dim

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(n_down):
            if self.spade_layers[0]:
                encoders.append(SPADEBlock(curr_dim, curr_dim*2, s_dim, kernel_size=k_size, Downsample=True))
            else:
                encoders.append(nn.Sequential(
                    nn.Conv2d(curr_dim, curr_dim*2, kernel_size=k_size, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim*2, affine=True),
                    nn.ReLU(inplace=True)
                ))
            self.num_channel[i + 1] = curr_dim * 2
            curr_dim = curr_dim * 2

        self.encoders = nn.Sequential(*encoders)

        # Bottleneck
        resnets = []
        for i in range(repeat_num // 2):
            if self.spade_layers[1]:
                resnets.append(SPADEResidualBlock(dim_in=curr_dim, dim_out=curr_dim, dim_c=s_dim))
            else:
                resnets.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            self.num_channel[i + 1 + n_down] = curr_dim

        for i in range(repeat_num - repeat_num // 2):
            if self.spade_layers[2]:
                resnets.append(SPADEResidualBlock(dim_in=curr_dim, dim_out=curr_dim, dim_c=s_dim))
            else:
                resnets.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            self.num_channel[i + 1 + n_down + repeat_num // 2] = curr_dim

        self.resnets = nn.Sequential(*resnets)

        # Up-Sampling
        decoders = []
        skippers = []
        for i in range(n_down):
            if self.spade_layers[3]:
                decoders.append(SPADEBlock(curr_dim, curr_dim//2, s_dim, kernel_size=k_size, Downsample=False))
            else:
                decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=2, padding=1, output_padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim//2, affine=True),
                    nn.ReLU(inplace=True)
                ))

            skippers.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim//2, kernel_size=k_size, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim//2, affine=True),
                nn.ReLU(inplace=True)
            ))

            curr_dim = curr_dim // 2

        self.decoders = nn.Sequential(*decoders)
        self.skippers = nn.Sequential(*skippers)

        if on_obj:
            layers = []
            layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
            layers.append(nn.Tanh())
            self.img_reg = nn.Sequential(*layers)

        else:
            layers = []
            layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
            layers.append(nn.Tanh())
            self.img_reg = nn.Sequential(*layers)

            layers = []
            layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
            layers.append(nn.Sigmoid())
            self.attetion_reg_hand = nn.Sequential(*layers)

            layers = []
            layers.append(nn.Conv2d(2 * curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
            layers.append(nn.Sigmoid())
            self.attetion_reg_bg = nn.Sequential(*layers)

    def inference(self, x, seg=None):
        # encoder, 0, 1, 2, 3 -> [256, 128, 64, 32]
        encoder_outs = self.encode(x, seg)

        # resnet, 32
        resnet_outs = []
        src_x = encoder_outs[-1]
        for i in range(self.repeat_num // 2):
            if self.spade_layers[1]:
                src_x = self.resnets[i](src_x, seg)
            else:
                src_x = self.resnets[i](src_x)
            resnet_outs.append(src_x)
        for i in range(self.repeat_num // 2, self.repeat_num):
            if self.spade_layers[2]:
                src_x = self.resnets[i](src_x, seg)
            else:
                src_x = self.resnets[i](src_x)
            resnet_outs.append(src_x)

        return encoder_outs, resnet_outs

    def forward(self, x, seg=None):

        # encoder, 0, 1, 2, 3 -> [256, 128, 64, 32]
        encoder_outs = self.encode(x, seg)

        # resnet, 32
        resnet_outs = encoder_outs[-1]
        for i in range(self.repeat_num // 2):
            if self.spade_layers[1]:
                resnet_outs = self.resnets[i](resnet_outs, seg)
            else:
                resnet_outs = self.resnets[i](resnet_outs)

        for i in range(self.repeat_num // 2, self.repeat_num):
            if self.spade_layers[2]:
                resnet_outs = self.resnets[i](resnet_outs, seg)
            else:
                resnet_outs = self.resnets[i](resnet_outs)

        # decoder, 0, 1, 2 -> [64, 128, 256]
        d_out = self.decode(resnet_outs, encoder_outs, seg)

        return d_out

    def encode(self, x, seg=None):
        e_out = self.encoders[0](x)

        encoder_outs = [e_out]
        for i in range(1, self.n_down + 1):
            if self.spade_layers[0]:
                e_out = self.encoders[i](e_out, seg)
            else:
                e_out = self.encoders[i](e_out)
            encoder_outs.append(e_out)
            #print(i, e_out.shape)
        return encoder_outs

    def decode(self, x, encoder_outs, seg=None):
        d_out = x
        for i in range(self.n_down):
            if self.spade_layers[3]:
                d_out = self.decoders[i](d_out, seg)  # x * 2
            else:
                d_out = self.decoders[i](d_out)  # x * 2
            skip = encoder_outs[self.n_down - 1 - i]
            d_out = torch.cat([skip, d_out], dim=1)
            d_out = self.skippers[i](d_out)
            # print(i, d_out.shape)
        return d_out

    def regress(self, x, y=None):
        if self.on_obj:
            return self.img_reg(x)
        else:
            return self.img_reg(x), self.attetion_reg_hand(x), self.attetion_reg_bg(torch.cat([x, y], dim=1))


class Generator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, bg_dim, img_dim, obj_dim, img_cond_dim=0, obj_cond_dim=0, conv_dim=64, repeat_num=6, spade_layers=[0, 0, 0, 0], attn_layers=[]):
        super(Generator, self).__init__()
        self._name = 'generator'
        print(spade_layers, attn_layers)

        self.n_down = 3
        self.repeat_num = repeat_num
        self.spade_layers = spade_layers
        self.attn_layers = attn_layers
        # background generator
        self.bg_model = ResNetGenerator(conv_dim=conv_dim, c_dim=bg_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down)

        # source generator
        self.obj_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=obj_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down, s_dim=obj_cond_dim, spade_layers=spade_layers, on_obj=True)

        # source generator
        self.src_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=img_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down, s_dim=img_cond_dim, spade_layers=spade_layers)

        # transfer generator
        self.tsf_model = ResUnetGenerator(conv_dim=conv_dim, c_dim=img_dim, repeat_num=repeat_num, k_size=3, n_down=self.n_down, s_dim=img_cond_dim, spade_layers=spade_layers)

        # attention sample
        if len(self.attn_layers) != 0:
            for attn_layer in self.attn_layers:
                attn = ExtractorAttn(self.src_model.num_channel[attn_layer], kernel_size=5, nonlinearity=nn.LeakyReLU(), softmax=True)
                setattr(self, 'attn_%d' % attn_layer, attn)

    def forward(self, bg_inputs, src_obj_inputs, tsf_obj_inputs, src_hand_inputs, tsf_hand_inputs, T,
                src_obj_conds=None, src_hand_conds=None, tsf_obj_conds=None, tsf_hand_conds=None,
                src_armask=None, tsf_armask=None):

        if src_obj_conds is None or src_hand_conds is None:
            src_bg_inputs = torch.cat([bg_inputs, src_obj_inputs[:, 3:]], dim=1)
        else:
            src_bg_inputs = torch.cat([bg_inputs, src_hand_conds], dim=1)

        if tsf_obj_conds is None or tsf_hand_conds is None:
            tsf_bg_inputs = torch.cat([bg_inputs, tsf_hand_inputs[:, 3:]], dim=1)
        else:
            tsf_bg_inputs = torch.cat([bg_inputs, tsf_hand_conds], dim=1)

        if src_armask is not None:
            src_bg_inputs = torch.cat([src_bg_inputs, src_armask], dim=1)

        if tsf_armask is not None:
            tsf_bg_inputs = torch.cat([tsf_bg_inputs, tsf_armask], dim=1)

        src_img_bg = self.bg_model(src_bg_inputs)

        tsf_img_bg = self.bg_model(tsf_bg_inputs)

        src_obj, src_hand, src_mask_bg, src_mask_hand, tsf_obj, tsf_hand, tsf_mask_bg, tsf_mask_hand = \
            self.infer_front(src_obj_inputs, tsf_obj_inputs, src_hand_inputs, tsf_hand_inputs, T,
                             src_obj_conds, src_hand_conds, tsf_obj_conds, tsf_hand_conds)

        # print(front_rgb.shape, front_mask.shape)
        return src_img_bg, tsf_img_bg, src_obj, src_hand, src_mask_bg, src_mask_hand, tsf_obj, tsf_hand, tsf_mask_bg, tsf_mask_hand


    def infer_front(self, src_obj_inputs, tsf_obj_inputs, src_hand_inputs, tsf_hand_inputs, T,
                    src_obj_conds, src_hand_conds, tsf_obj_conds, tsf_hand_conds):
        # encoder
        src_x = self.src_model.encoders[0](src_hand_inputs)
        tsf_x = self.tsf_model.encoders[0](tsf_hand_inputs)

        src_encoder_outs = [src_x]
        tsf_encoder_outs = [tsf_x]
        for i in range(1, self.n_down + 1):
            if self.spade_layers[0]:
                src_x = self.src_model.encoders[i](src_x, src_hand_conds)
            else:
                src_x = self.src_model.encoders[i](src_x)

            if self.spade_layers[0]:
                tsf_x = self.tsf_model.encoders[i](tsf_x, tsf_hand_conds)
            else:
                tsf_x = self.tsf_model.encoders[i](tsf_x)

            layer_num = i
            if layer_num in self.attn_layers:
                attn = getattr(self, 'attn_%d' % layer_num)
                warp = self.transform(src_x, T, y=tsf_x, attn=attn)
            else:
                warp = self.transform(src_x, T)
            tsf_x = tsf_x + warp

            src_encoder_outs.append(src_x)
            tsf_encoder_outs.append(tsf_x)

        # resnets
        for i in range(self.repeat_num // 2):
            if self.spade_layers[1]:
                src_x = self.src_model.resnets[i](src_x, src_hand_conds)
            else:
                src_x = self.src_model.resnets[i](src_x)

            if self.spade_layers[1]:
                tsf_x = self.tsf_model.resnets[i](tsf_x, tsf_hand_conds)
            else:
                tsf_x = self.tsf_model.resnets[i](tsf_x)

            layer_num = i + self.n_down + 1
            if layer_num in self.attn_layers:
                attn = getattr(self, 'attn_%d' % layer_num)
                warp = self.transform(src_x, T, y=tsf_x, attn=attn)
            else:
                warp = self.transform(src_x, T)
            tsf_x = tsf_x + warp

        for i in range(self.repeat_num // 2, self.repeat_num):
            if self.spade_layers[2]:
                src_x = self.src_model.resnets[i](src_x, src_hand_conds)
            else:
                src_x = self.src_model.resnets[i](src_x)

            if self.spade_layers[2]:
                tsf_x = self.tsf_model.resnets[i](tsf_x, tsf_hand_conds)
            else:
                tsf_x = self.tsf_model.resnets[i](tsf_x)

            layer_num = i + self.n_down + 1
            if layer_num in self.attn_layers:
                attn = getattr(self, 'attn_%d' % layer_num)
                warp = self.transform(src_x, T, y=tsf_x, attn=attn)
            else:
                warp = self.transform(src_x, T)
            tsf_x = tsf_x + warp

        # decoders
        src_y = self.obj_model(src_obj_inputs, src_obj_conds)
        tsf_y = self.obj_model(tsf_obj_inputs, tsf_obj_conds)
        if self.spade_layers[3]:
            src_x = self.src_model.decode(src_x, src_encoder_outs, src_hand_conds)
            tsf_x = self.tsf_model.decode(tsf_x, tsf_encoder_outs, tsf_hand_conds)
        else:
            src_x = self.src_model.decode(src_x, src_encoder_outs)
            tsf_x = self.tsf_model.decode(tsf_x, tsf_encoder_outs)
        src_hand, src_mask_hand, src_mask_bg = self.src_model.regress(src_x, src_y)
        tsf_hand, tsf_mask_hand, tsf_mask_bg = self.tsf_model.regress(tsf_x, tsf_y)

        src_obj = self.obj_model.regress(src_y)
        tsf_obj = self.obj_model.regress(tsf_y)

        # print(front_rgb.shape, front_mask.shape)
        return src_obj, src_hand, src_mask_bg, src_mask_hand, tsf_obj, tsf_hand, tsf_mask_bg, tsf_mask_hand

    def resize_trans(self, x, T):
        _, _, h, w = x.shape

        T_scale = T.permute(0, 3, 1, 2)  # (bs, 2, h, w)
        T_scale = F.interpolate(T_scale, size=(h, h), mode='bilinear', align_corners=True)
        T_scale = T_scale.permute(0, 2, 3, 1)  # (bs, h, w, 2)

        return T_scale

    def stn(self, x, T):
        x_trans = F.grid_sample(x, T)

        return x_trans

    def transform(self, x, T, y=None, attn=None):
        T_scale = self.resize_trans(x, T)
        if attn is not None:
            # identity grid
            _x = torch.arange(start=-1.0, end=1.0, step=2.0 / x.shape[2])
            _y = torch.arange(start=-1.0, end=1.0, step=2.0 / x.shape[2])
            xx, yy = torch.meshgrid(_x, _y)
            idt = torch.stack([xx, yy], dim=2).unsqueeze(dim=0).cuda()
            x_trans = attn(x, y, (T_scale - idt).permute(0, 3, 1, 2))
        else:
            x_trans = self.stn(x, T_scale)
        return x_trans
