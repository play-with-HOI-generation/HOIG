import torch
import torch.nn.functional as F
from collections import OrderedDict
import utils.util as util
from models.base_model import BaseModel
from models.networks import NetworksFactory, HandModelRecovery
from models.networks.vgg19 import Vgg19, VGGLoss
from utils.nmr import MANORenderer


OBJNAMES = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
            '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
            '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
            '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']


class HandRecoveryFlow(torch.nn.Module):

    def __init__(self, opt):
        super(HandRecoveryFlow, self).__init__()
        self._name = 'HandRecoveryFlow'
        self._opt = opt

        # create networks
        self._init_create_networks()

    def _create_hmr(self):
        hmr = HandModelRecovery(mano_path=self._opt.mano_model)
        # saved_data = torch.load(self._opt.hmr_model)
        # hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self):
        render = MANORenderer(map_name=self._opt.map_name,
                              uv_map_path=self._opt.uv_mapping,
                              tex_size=self._opt.tex_size,
                              image_size=self._opt.image_size, fill_back=False,
                              anti_aliasing=True, background_color=(0, 0, 0),
                              has_front=False)

        return render

    def _init_create_networks(self):
        # hmr and render
        self._hmr = self._create_hmr()
        self._render = self._create_render()

    def forward(self, src_img, ref_img, src_mano, ref_mano):
        # get MANO information
        src_info = self._hmr.get_details(src_mano)
        ref_info = self._hmr.get_details(ref_mano)

        # process source inputs
        src_crop_mask_hand_list = []
        ref_crop_mask_hand_list = []
        src_cond_list = []
        ref_cond_list = []
        src_seg_list = []
        ref_seg_list = []
        render_img_src_list = []
        render_img_ref_list = []
        T_hand_list = []
        
        bs = len(src_info['objName'])
        for i in range(bs):
            objname = OBJNAMES[src_info['objName'][i].item()]
            length = getattr(self._render, 'faces_{}'.format(objname)).max() + 1
            src_f2verts, src_fim, src_wim = self._render.render_fim_wim(src_info['cam'][i:i+1], src_info['verts'][i:i+1, :length], objname)
            src_f2verts = src_f2verts[:, :, :, 0:2]
            src_f2verts[:, :, :, 1] *= -1
            src_cond, _ = self._render.encode_fim(src_info['cam'][i:i+1], src_info['verts'][i:i+1, :length], objname, fim=src_fim, transpose=True)
            src_seg, _ = self._render.encode_sem(src_info['cam'], src_info['verts'], objname, fim=src_fim, transpose=True)
            src_seg = torch.cat([(src_seg == i).float() for i in range(1, 16)], dim=1)
            src_crop_mask_hand = util.morph(1 - ((src_fim != -1) & (src_fim < 1538))[:, None].float(), ks=3, mode='erode')

            _, ref_fim, ref_wim = self._render.render_fim_wim(ref_info['cam'][i:i+1], ref_info['verts'][i:i+1, :length], objname)
            ref_cond, _ = self._render.encode_fim(ref_info['cam'], ref_info['verts'][i:i+1, :length], objname, fim=ref_fim, transpose=True)
            ref_seg, _ = self._render.encode_sem(ref_info['cam'], ref_info['verts'], objname, fim=ref_fim, transpose=True)
            ref_seg = torch.cat([(ref_seg == i).float() for i in range(1, 16)], dim=1)
            ref_crop_mask_hand = util.morph(1 - ((ref_fim != -1) & (ref_fim < 1538))[:, None].float(), ks=3, mode='erode')

            T, O = self._render.cal_bc_transform(src_f2verts, src_fim, ref_fim, ref_wim)
            T_hand = T * (ref_crop_mask_hand[:,0][:,:,:,None] == 0) + (-2) * torch.ones_like(T) * (ref_crop_mask_hand[:,0][:,:,:,None] == 1)

            input_texture = self._render.get_texture_backward_warp(src_img[i:i+1], src_f2verts, src_fim, objname)
            T_ref = self._render.sample_from_texture_dense(ref_fim, ref_wim, objname)
            render_img_ref = F.grid_sample(input_texture, T_ref, align_corners=True)
            T_src = self._render.sample_from_texture_dense(src_fim, src_wim, objname)
            render_img_src = F.grid_sample(input_texture, T_src, align_corners=True)

            src_crop_mask_hand_list.append(src_crop_mask_hand)
            ref_crop_mask_hand_list.append(ref_crop_mask_hand)
            src_cond_list.append(src_cond)
            ref_cond_list.append(ref_cond)
            src_seg_list.append(src_seg)
            ref_seg_list.append(ref_seg)
            render_img_src_list.append(render_img_src)
            render_img_ref_list.append(render_img_ref)
            T_hand_list.append(T_hand)

        src_crop_mask_hand = torch.cat(src_crop_mask_hand_list, dim=0)
        ref_crop_mask_hand = torch.cat(ref_crop_mask_hand_list, dim=0)
        src_cond = torch.cat(src_cond_list, dim=0)
        ref_cond = torch.cat(ref_cond_list, dim=0)
        src_seg = torch.cat(src_seg_list, dim=0)
        ref_seg = torch.cat(ref_seg_list, dim=0)
        render_img_src = torch.cat(render_img_src_list, dim=0)
        render_img_ref = torch.cat(render_img_ref_list, dim=0)
        T_hand = torch.cat(T_hand_list, dim=0)
        # masks
        src_crop_mask_bg = util.morph(src_cond[:, -1:, :, :], ks=3, mode='erode')
        ref_crop_mask_bg = util.morph(ref_cond[:, -1:, :, :], ks=3, mode='erode')

        src_cond_hand_mask = (src_cond[:, :1] < 1.5).float()
        src_cond_hand = torch.cat([src_cond_hand_mask * src_cond[:, :2],
                                   src_cond[:, 2:] + 1 - src_cond_hand_mask], dim=1)
        src_cond_obj_mask = (src_cond[:, :1] > 1.5).float()
        src_cond_obj = torch.cat([src_cond_obj_mask * src_cond[:, :2],
                                  src_cond[:, 2:] + 1 - src_cond_obj_mask], dim=1)

        ref_cond_hand_mask = (ref_cond[:, :1] < 1.5).float()
        ref_cond_hand = torch.cat([ref_cond_hand_mask * ref_cond[:, :2],
                                   ref_cond[:, 2:] + 1 - ref_cond_hand_mask], dim=1)
        ref_cond_obj_mask = (ref_cond[:, :1] > 1.5).float()
        ref_cond_obj = torch.cat([ref_cond_obj_mask * ref_cond[:, :2],
                                   ref_cond[:, 2:] + 1 - ref_cond_obj_mask], dim=1)

        # src input
        input_G_src_obj = torch.cat([render_img_src * (src_crop_mask_hand - src_crop_mask_bg), src_cond_obj, src_seg[:, 6:]], dim=1)
        input_G_src_hand = torch.cat([src_img * (1 - src_crop_mask_hand), src_cond_hand, src_seg[:, :6]], dim=1)

        # tsf input
        input_G_tsf_obj = torch.cat([render_img_ref * (ref_crop_mask_hand - ref_crop_mask_bg), ref_cond_obj, ref_seg[:, 6:]], dim=1)
        input_G_ref_hand = torch.cat([render_img_ref * (1 - ref_crop_mask_hand), ref_cond_hand, ref_seg[:, :6]], dim=1)

        # bg input
        src_bg_mask = util.morph(src_cond[:, -1:, :, :], ks=15, mode='erode')
        input_G_src_bg = torch.cat([src_img * src_bg_mask, src_bg_mask], dim=1)

        if self._opt.bg_both:
            ref_bg_mask = util.morph(ref_cond[:, -1:, :, :], ks=15, mode='erode')
            input_G_tsf_bg = torch.cat([ref_img * ref_bg_mask, ref_bg_mask], dim=1)
        else:
            input_G_tsf_bg = None

        return input_G_src_bg, input_G_tsf_bg, input_G_src_obj, input_G_tsf_obj, input_G_src_hand, input_G_ref_hand, \
               T_hand, src_crop_mask_bg, ref_crop_mask_bg, src_crop_mask_hand, ref_crop_mask_hand, None


    def cal_hand_bbox(self, kps, factor=1.2):
        """
        Args:
            kps (torch.cuda.FloatTensor): (N, 19, 2)
            factor (float):

        Returns:
            bbox: (N, 4)
        """
        image_size = self._opt.image_size
        bs = kps.shape[0]
        kps = (kps + 1) / 2.0
        zeros = torch.zeros((bs,), device=kps.device)
        ones = torch.ones((bs,), device=kps.device)

        min_x, _ = torch.min(kps[:, :, 0], dim=1)
        max_x, _ = torch.max(kps[:, :, 0], dim=1)
        middle_x = (min_x + max_x) / 2
        width = (max_x - min_x) * factor
        min_x = torch.max(zeros, middle_x - width / 2)
        max_x = torch.min(ones, middle_x + width / 2)

        min_y, _ = torch.min(kps[:, :, 1], dim=1)
        max_y, _ = torch.max(kps[:, :, 1], dim=1)
        middle_y = (min_y + max_y) / 2
        height = (max_y - min_y) * factor
        min_y = torch.max(zeros, middle_y - height / 2)
        max_y = torch.min(ones, middle_y + height / 2)

        min_x = (min_x * image_size).long()  # (T,)
        max_x = (max_x * image_size).long()  # (T,)
        min_y = (min_y * image_size).long()  # (T,)
        max_y = (max_y * image_size).long()  # (T,)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        bboxs = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        return bboxs


class Trainer(BaseModel):
    def __init__(self, opt, use_ddp=False):
        super(Trainer, self).__init__(opt, use_ddp)
        self._name = 'Trainer'

        # set device
        if use_ddp:
            self.device = torch.device('cuda:{}'.format(opt.local_rank))

        # create networks
        self._init_create_networks(use_ddp=use_ddp)

        # init train variables and losses
        if self._is_train:
            self._init_train_vars()
            self._init_losses(use_ddp=use_ddp)

        # load networks and optimizers
        if self._opt.load_path != 'None':
            # ipdb.set_trace()
            self._load_params(self._G, self._opt.load_path, need_module=len(self._gpu_ids) > 1)
        elif not self._is_train or self._opt.load_epoch > 0:
            self.load()

        self.colorize = util.Colorize(n=16)

        # prefetch variables
        self._init_prefetch_inputs()

    def _init_create_networks(self, use_ddp=False):
        multi_gpus = len(self._gpu_ids) > 1

        # hand recovery Flow
        self._hdr = HandRecoveryFlow(opt=self._opt)
        if use_ddp:
            self._hdr.eval()
            self._hdr.cuda()
            self._hdr.to(self.device)
        else:
            if multi_gpus:
                self._hdr = torch.nn.DataParallel(self._hdr)
            self._hdr.eval()
            self._hdr.cuda()

        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if use_ddp:
            self._G.cuda()
            self._G = torch.nn.parallel.DistributedDataParallel(self._G, device_ids=[self._opt.local_rank],
                                                                output_device=self._opt.local_rank,
                                                                find_unused_parameters=True)
        else:
            if multi_gpus:
                self._G = torch.nn.DataParallel(self._G)
            self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if use_ddp:
            self._D.cuda()
            self._D = torch.nn.parallel.DistributedDataParallel(self._D, device_ids=[self._opt.local_rank],
                                                                output_device=self._opt.local_rank,
                                                                find_unused_parameters=True)
        else:
            if multi_gpus:
                self._D = torch.nn.DataParallel(self._D)
            self._D.cuda()

    def _create_generator(self):
        if self._opt.use_spade:
            return NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=13, img_dim=3, obj_dim=3, img_cond_dim=9,
                                               obj_cond_dim=12, repeat_num=self._opt.repeat_num)
        else:
            return NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=13, img_dim=12, obj_dim=12,
                                               repeat_num=self._opt.repeat_num)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_patch_gan', input_nc=24,
                                           norm_type=self._opt.norm_type, ndf=64, n_layers=4, use_sigmoid=False)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=(self._opt.G_adam_b1, self._opt.G_adam_b2))
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=(self._opt.D_adam_b1, self._opt.D_adam_b2))

    def _init_prefetch_inputs(self):
        self._real_src = None
        self._real_tsf = None
        self._real_tex = None
        self._bg_mask = None
        self._input_src = None
        self._input_G_bg = None
        self._input_G_src = None
        self._input_G_tsf = None
        self._T_src = None
        self._T_ref = None

    def _init_losses(self, use_ddp=False):
        # define loss functions
        multi_gpus = len(self._gpu_ids) > 1
        self._crt_l1 = torch.nn.L1Loss()

        if self._opt.mask_bce:
            self._crt_mask = torch.nn.BCELoss()
        else:
            self._crt_mask = torch.nn.MSELoss()

        vgg_net = Vgg19()
        if self._opt.use_vgg:
            self._crt_tsf = VGGLoss(vgg=vgg_net)
            if use_ddp:
                self._crt_tsf.to(self.device)
            else:
                if multi_gpus:
                    self._crt_tsf = torch.nn.DataParallel(self._crt_tsf)
                self._crt_tsf.cuda()

        # init losses G
        self._loss_g_rec = self._Tensor([0])
        self._loss_g_tsf = self._Tensor([0])
        self._loss_g_adv = self._Tensor([0])
        self._loss_g_smooth = self._Tensor([0])
        self._loss_g_mask = self._Tensor([0])
        self._loss_g_mask_smooth = self._Tensor([0])

        # init losses D
        self._d_real = self._Tensor([0])
        self._d_fake = self._Tensor([0])

    def set_input(self, input):

        with torch.no_grad():
            imageA = input['imageA']
            imageB = input['imageB']
            manoA = input['manoA']
            manoB = input['manoB']

            src_img = imageA.cuda()
            tsf_img = imageB.cuda()

            src_mano = {key: manoA[key].cuda() for key in list(manoA.keys())}
            tsf_mano = {key: manoB[key].cuda() for key in list(manoB.keys())}

            input_G_src_bg, input_G_tsf_bg, input_G_src_obj, input_G_tsf_obj, input_G_src_hand, input_G_tsf_hand, \
            T, src_crop_mask_bg, tsf_crop_mask_bg, src_crop_mask_hand, tsf_crop_mask_hand, hand_bbox \
                = self._hdr(src_img, tsf_img, src_mano, tsf_mano)

            self._real_src = src_img
            self._real_tsf = tsf_img

            self._bg_mask = torch.cat((src_crop_mask_bg, tsf_crop_mask_bg), dim=0)
            self._hand_mask = torch.cat((src_crop_mask_hand, tsf_crop_mask_hand), dim=0)
            if self._opt.bg_both:
                self._input_G_bg = torch.cat([input_G_src_bg, input_G_tsf_bg], dim=0)
            else:
                self._input_G_bg = input_G_src_bg
            self._input_G_src_obj = input_G_src_obj
            self._input_G_tsf_obj = input_G_tsf_obj
            self._input_G_src_hand = input_G_src_hand
            self._input_G_tsf_hand = input_G_tsf_hand
            self._T = T
            self._hand_bbox = hand_bbox

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        # generate fake images

        if self._opt.use_spade:
            _input_G_src_obj_rgb = self._input_G_src_obj[:, :3]
            _input_G_src_obj_cond = self._input_G_src_obj[:, 3:]
            _input_G_src_hand_rgb = self._input_G_src_hand[:, :3]
            _input_G_src_hand_cond = self._input_G_src_hand[:, 3:]

            _input_G_tsf_obj_rgb = self._input_G_tsf_obj[:, :3]
            _input_G_tsf_obj_cond = self._input_G_tsf_obj[:, 3:]
            _input_G_tsf_hand_rgb = self._input_G_tsf_hand[:, :3]
            _input_G_tsf_hand_cond = self._input_G_tsf_hand[:, 3:]

            fake_src_bg, fake_tsf_bg, fake_src_obj, fake_src_hand, fake_src_mask_bg, fake_src_mask_hand, \
            fake_tsf_obj, fake_tsf_hand, fake_tsf_mask_bg, fake_tsf_mask_hand = \
                self._G.forward(self._input_G_bg, _input_G_src_obj_rgb, _input_G_tsf_obj_rgb,
                                _input_G_src_hand_rgb, _input_G_tsf_hand_rgb, T=self._T,
                                src_obj_conds=_input_G_src_obj_cond, src_hand_conds=_input_G_src_hand_cond,
                                tsf_obj_conds=_input_G_tsf_obj_cond, tsf_hand_conds=_input_G_tsf_hand_cond)
        else:
            fake_src_bg, fake_tsf_bg, fake_src_obj, fake_src_hand, fake_src_mask_bg, fake_src_mask_hand, \
            fake_tsf_obj, fake_tsf_hand, fake_tsf_mask_bg, fake_tsf_mask_hand = \
                self._G.forward(self._input_G_bg, self._input_G_src_obj, self._input_G_src_hand,
                                self._input_G_tsf_obj, self._input_G_tsf_hand, T=self._T)

        fake_src_imgs = fake_src_mask_bg * fake_src_bg + (1 - fake_src_mask_bg) * (fake_src_obj * fake_src_mask_hand + fake_src_hand * (1 - fake_src_mask_hand))
        fake_tsf_imgs = fake_tsf_mask_bg * fake_tsf_bg + (1 - fake_tsf_mask_bg) * (fake_tsf_obj * fake_tsf_mask_hand + fake_tsf_hand * (1 - fake_tsf_mask_hand))

        fake_masks_bg = torch.cat([fake_src_mask_bg, fake_tsf_mask_bg], dim=0)
        fake_masks_hand = torch.cat([fake_src_mask_hand, fake_tsf_mask_hand], dim=0)
        fake_src_fg = - fake_src_mask_bg + (1 - fake_src_mask_bg) * (fake_src_obj * fake_src_mask_hand + fake_src_hand * (1 - fake_src_mask_hand))
        fake_tsf_fg = - fake_tsf_mask_bg + (1 - fake_tsf_mask_bg) * (fake_tsf_obj * fake_tsf_mask_hand + fake_tsf_hand * (1 - fake_tsf_mask_hand))

        src_seg = torch.cat([self._input_G_src_hand[:, 6:], self._input_G_src_obj[:, 3:]], dim=1)
        tsf_seg = torch.cat([self._input_G_tsf_hand[:, 6:], self._input_G_tsf_obj[:, 3:]], dim=1)

        # keep data for visualization
        if keep_data_for_visuals:
            self.visual_imgs(fake_src_bg, fake_tsf_bg, fake_src_imgs, fake_tsf_imgs, fake_src_fg, fake_tsf_fg, fake_masks_bg, fake_masks_hand, src_seg, tsf_seg)

        return fake_src_bg, fake_tsf_bg, fake_src_imgs, fake_tsf_imgs, fake_masks_bg, fake_masks_hand

    def optimize_parameters(self, trainable=True, keep_data_for_visuals=False):
        if self._is_train:

            # run inference
            fake_src_bg, fake_tsf_bg, fake_src_imgs, fake_tsf_imgs, fake_masks_bg, fake_masks_hand = self.forward(keep_data_for_visuals=keep_data_for_visuals)

            loss_G = self._optimize_G(fake_src_imgs, fake_tsf_imgs, fake_masks_bg, fake_masks_hand)

            self._optimizer_G.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()

            # train D
            if trainable:
                loss_D = self._optimize_D(fake_tsf_imgs)
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

    def _optimize_G(self, fake_src_imgs, fake_tsf_imgs, fake_masks_bg, fake_masks_hand):
        fake_input_D = torch.cat([fake_tsf_imgs, self._input_G_tsf_obj[:, 3:], self._input_G_tsf_hand[:, 3:]], dim=1)
        d_fake_outs = self._D.forward(fake_input_D)
        self._loss_g_adv = self._compute_loss_D(d_fake_outs, 0) * self._opt.lambda_D_prob

        self._loss_g_rec = self._crt_l1(fake_src_imgs, self._real_src) * self._opt.lambda_rec

        if self._opt.use_vgg:
            self._loss_g_tsf = torch.mean(self._crt_tsf(fake_tsf_imgs, self._real_tsf)) * self._opt.lambda_tsf
        else:
            self._loss_g_tsf = torch.mean(self._crt_tsf(fake_tsf_imgs, self._real_tsf)) * self._opt.lambda_tsf

        # loss mask
        self._loss_g_mask = (self._crt_mask(fake_masks_bg, self._bg_mask) + self._crt_mask(fake_masks_hand, self._hand_mask)) \
                            * self._opt.lambda_mask

        if self._opt.lambda_mask_smooth != 0:
            self._loss_g_mask_smooth = (self._compute_loss_smooth(fake_masks_bg) + self._compute_loss_smooth(fake_masks_hand)) \
                                       * self._opt.lambda_mask_smooth

        # combine losses
        return self._loss_g_adv + self._loss_g_rec + self._loss_g_tsf + self._loss_g_mask + self._loss_g_mask_smooth

    def _optimize_D(self, fake_tsf_imgs):
        tsf_cond = torch.cat([self._input_G_tsf_obj[:, 3:], self._input_G_tsf_hand[:, 3:]], dim=1)
        fake_input_D = torch.cat([fake_tsf_imgs.detach(), tsf_cond], dim=1)
        real_input_D = torch.cat([self._real_tsf, tsf_cond], dim=1)

        d_real_outs = self._D.forward(real_input_D)
        d_fake_outs = self._D.forward(fake_input_D)

        _loss_d_real = self._compute_loss_D(d_real_outs, 1) * self._opt.lambda_D_prob
        _loss_d_fake = self._compute_loss_D(d_fake_outs, -1) * self._opt.lambda_D_prob

        self._d_real = torch.mean(d_real_outs)
        self._d_fake = torch.mean(d_fake_outs)

        # combine losses
        return _loss_d_real + _loss_d_fake

    def _compute_loss_D(self, x, y):
        return torch.mean((x - y) ** 2)

    def _compute_loss_smooth(self, mat):
        return torch.mean(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.mean(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_rec', self._loss_g_rec.item()),
                                 ('g_tsf', self._loss_g_tsf.item()),
                                 ('g_adv', self._loss_g_adv.item()),
                                 ('g_mask', self._loss_g_mask.item()),
                                 ('g_mask_smooth', self._loss_g_mask_smooth.item()),
                                 ('d_real', self._d_real.item()),
                                 ('d_fake', self._d_fake.item())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        # inputs
        visuals['1_real_img'] = self._vis_input
        visuals['2_input_src_obj'] = self._vis_src_obj
        visuals['2_input_src_hand'] = self._vis_src_hand
        visuals['2_input_tsf_obj'] = self._vis_tsf_obj
        visuals['2_input_tsf_hand'] = self._vis_tsf_hand
        visuals['3_fake_src_bg'] = self._vis_fake_src_bg
        visuals['4_fake_tsf_bg'] = self._vis_fake_tsf_bg
        visuals['5_fake_src_color'] = self._vis_fake_src_color
        visuals['6_fake_tsf_color'] = self._vis_fake_tsf_color
        visuals['7_src_seg'] = self._vis_src_seg
        visuals['8_ref_seg'] = self._vis_ref_seg

        # outputs
        visuals['10_fake_tsf'] = self._vis_fake_tsf
        visuals['11_fake_src'] = self._vis_fake_src
        visuals['12_fake_mask_bg'] = self._vis_mask_bg
        visuals['13_fake_mask_hand'] = self._vis_mask_hand

        # batch outputs
        visuals['14_batch_real_img'] = self._vis_batch_real
        visuals['15_batch_fake_img'] = self._vis_batch_fake
        visuals['16_batch_src_img'] = self._vis_batch_src

        return visuals

    @torch.no_grad()
    def visual_imgs(self, fake_src_bg, fake_tsf_bg, fake_src_imgs, fake_tsf_imgs, fake_src_fg, fake_tsf_fg, fake_masks_bg, fake_masks_hand, src_seg, ref_seg):
        ids = fake_masks_bg.shape[0] // 2
        self._vis_input = util.tensor2im(self._real_src)
        self._vis_src_obj = util.tensor2im(self._input_G_src_obj[0, 0:3])
        self._vis_src_hand = util.tensor2im(self._input_G_src_hand[0, 0:3])
        self._vis_tsf_obj = util.tensor2im(self._input_G_tsf_obj[0, 0:3])
        self._vis_tsf_hand = util.tensor2im(self._input_G_tsf_hand[0, 0:3])
        self._vis_fake_src_bg = util.tensor2im(fake_src_bg)
        self._vis_fake_tsf_bg = util.tensor2im(fake_tsf_bg)
        self._vis_fake_src_color = util.tensor2im(fake_src_fg)
        self._vis_fake_tsf_color = util.tensor2im(fake_tsf_fg)
        self._vis_fake_src = util.tensor2im(fake_src_imgs)
        self._vis_fake_tsf = util.tensor2im(fake_tsf_imgs)
        self._vis_mask_bg = util.tensor2maskim(fake_masks_bg[ids])
        self._vis_mask_hand = util.tensor2maskim(fake_masks_hand[ids])

        src_seg_vis = (src_seg.sum(dim=1, keepdim=True) != 0) * (src_seg.argmax(dim=1, keepdim=True) + 1)
        ref_seg_vis = (ref_seg.sum(dim=1, keepdim=True) != 0) * (ref_seg.argmax(dim=1, keepdim=True) + 1)
        self._vis_src_seg = util.tensor2im(torch.stack([self.colorize.label2color(src_seg_vis[i]) for i in range(src_seg_vis.shape[0])], dim=0))
        self._vis_ref_seg = util.tensor2im(torch.stack([self.colorize.label2color(ref_seg_vis[i]) for i in range(ref_seg_vis.shape[0])], dim=0))

        self._vis_batch_real = util.tensor2im(self._real_tsf, idx=-1)
        self._vis_batch_fake = util.tensor2im(fake_tsf_imgs, idx=-1)
        self._vis_batch_src = util.tensor2im(self._real_src, idx=-1)

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch, need_module=True)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch, need_module=True)

            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_D, 'D', load_epoch)
            if self._is_train and load_epoch > self._opt.nepochs_no_decay:
                for _ in range(self._opt.nepochs_no_decay, load_epoch):
                    self.update_learning_rate()

    def update_learning_rate(self):
        # updated learning rate G
        final_lr = self._opt.final_lr

        lr_decay_G = (self._opt.lr_G - final_lr) / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' % (self._current_lr_G + lr_decay_G, self._current_lr_G))

        # update learning rate D
        lr_decay_D = (self._opt.lr_D - final_lr) / self._opt.nepochs_decay
        self._current_lr_D -= lr_decay_D
        for param_group in self._optimizer_D.param_groups:
            param_group['lr'] = self._current_lr_D
        print('update D learning rate: %f -> %f' % (self._current_lr_D + lr_decay_D, self._current_lr_D))




