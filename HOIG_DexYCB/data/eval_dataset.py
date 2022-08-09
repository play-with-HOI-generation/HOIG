import numpy as np
import os
from data.dataset_base import DatasetBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import json
import utils.hand_utils as handutils
import cv2

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans

def augmentation(img, bbox):
    img = img.copy()
    trans, scale, rot, do_flip = [0, 0], 1.0, 0.0, False
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, (256, 256))
    return img, trans


class InterHandEvalDataset(DatasetBase):

    def __init__(self, opt, is_for_train=True):
        super(InterHandEvalDataset, self).__init__(opt, is_for_train)
        self._name = 'InterHandDataset'
        self.param_dir = os.path.join(opt.data_dir, opt.params_dir)
        self.pic_dir = os.path.join(opt.data_dir, opt.images_dir)
        self.data_split = opt.data_split

        if not os.path.exists(self.param_dir):
            raise ValueError("param_dir: %s not exist" % self.param_dir)
        if not os.path.exists(self.pic_dir):
            raise ValueError("pic_dir: %s not exist" % self.pic_dir)

        with open(os.path.join(self.param_dir, 'InterHand2.6M_train_MANO_NeuralAnnot.json')) as f:
            self.mano_params = json.load(f)
        with open(os.path.join(self.param_dir, 'InterHand2.6M_train_camera.json')) as f:
            self.cam_params = json.load(f)
        with open(os.path.join(self.param_dir, 'InterHand_Tiny_bbx.pkl'), 'rb') as f:
            self.bbx_params = pickle.load(f)

        _eval_pairs_dir = os.path.join(self.param_dir, 'eval_pairs_act_ind.pkl')
        with open(opt.eval_pairs, "rb") as fid:
            _eval_dict = pickle.load(fid, encoding='latin1')
        self._eval_list = [[src, tsf] for src in _eval_dict for tsf in _eval_dict[src]]
        self._num_videos = len(self._eval_list)

        self._create_transform()

    def __getitem__(self, index):
        src, tsf = self._eval_list[index % self._num_videos]
        capture_id, cam_id, action_id = src.split('/')[:3]
        src_frame = src.split('/')[-1]
        tsf_frame = tsf.split('/')[-1]

        image_a, mano_a = self._get_sample(capture_id, cam_id, action_id, src_frame)
        image_b, mano_b = self._get_sample(capture_id, cam_id, action_id, tsf_frame)

        image_a = self._transform(image_a)
        image_b = self._transform(image_b)

        return {'imageA': image_a, 'manoA': mano_a,
                'imageB': image_b, 'manoB': mano_b,
                'pathA': src, 'pathB': tsf}

    def _get_sample(self, capture_id, cam_id, action_id, frame_id):
        image = cv2.imread(os.path.join(self.pic_dir, capture_id, cam_id, action_id, frame_id))

        bbox = self.bbx_params[os.path.join(capture_id, cam_id, action_id)]
        image, trans = augmentation(image, bbox)

        image_inv = (image / 255.0)[:, :, ::-1].copy()

        theta = {}
        for hand_type in ['left', 'right']:
            mano_param = self.mano_params[capture_id[7:]][frame_id[5:-4]][hand_type]
            hand_np = np.array(mano_param['pose'] + mano_param['shape'] + mano_param['trans']).astype(np.float32)
            cam_np = np.array(self.cam_params[capture_id[7:]]['campos'][cam_id[3:]] +
                              self.cam_params[capture_id[7:]]['camrot'][cam_id[3:]][0] +
                              self.cam_params[capture_id[7:]]['camrot'][cam_id[3:]][1] +
                              self.cam_params[capture_id[7:]]['camrot'][cam_id[3:]][2] +
                              self.cam_params[capture_id[7:]]['focal'][cam_id[3:]] +
                              self.cam_params[capture_id[7:]]['princpt'][cam_id[3:]])
            theta_np = np.concatenate([cam_np, trans.reshape(-1), hand_np], axis=0)
            theta[hand_type] = torch.from_numpy(theta_np).float()

        return image_inv, theta

    def __len__(self):
        return self._num_videos * self._opt.num_repeats

    def _create_transform(self):
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self._transform = transforms.Compose(transform_list)
