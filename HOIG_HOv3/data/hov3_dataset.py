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


OBJNAMES = ['003_cracker_box', '004_sugar_box', '006_mustard_bottle', '010_potted_meat_can', '011_banana', '021_bleach_cleanser', '025_mug', '035_power_drill', '037_scissors']


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


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def read_annotation(base_dir, seq_name, file_id, split):
    meta_filename = os.path.join(base_dir, split, seq_name, 'meta', file_id + '.pkl')
    _assert_exist(meta_filename)
    pkl_data = load_pickle_data(meta_filename)
    return pkl_data

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2])-1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            #if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])


    for k, v in d.items():
        if k in ['v','vn','f','vt','ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


class HOv3Dataset(DatasetBase):

    def __init__(self, opt, is_for_train=True):
        super(HOv3Dataset, self).__init__(opt, is_for_train)
        self._name = 'HOv3Dataset'
        self.data_dir = opt.data_dir
        self.param_dir = os.path.join(opt.data_dir, opt.params_dir)
        self.pic_dir = os.path.join(opt.data_dir, opt.images_dir)
        self.data_split = 'train' if is_for_train else 'test'
        self.pairs_dir = opt.pairs_dir

        if not os.path.exists(self.param_dir):
            raise ValueError("param_dir: %s not exist" % self.param_dir)
        if not os.path.exists(self.pic_dir):
            raise ValueError("pic_dir: %s not exist" % self.pic_dir)

        with open(os.path.join(self.param_dir, 'HOv3-CR_bbx.pkl'), 'rb') as f:
            self.bbx_params = pickle.load(f)

        _vid_list_dir = os.path.join(self.param_dir, 'HOv3-CR_train_new.pkl' if is_for_train else 'HOv3-CR_test_new.pkl')
        with open(_vid_list_dir, 'rb') as f:
            self._vids_dict = pickle.load(f)

        if os.path.exists(self.pairs_dir):
            with open(self.pairs_dir, "rb") as f:
                self._pairs_list = pickle.load(f)
        else:
            self._pairs_list = None

        self._vids_list = list(self._vids_dict)
        self._num_videos = len(self._vids_list) if self._pairs_list is None else len(self._pairs_list)

        self._create_transform()

    def __getitem__(self, index):
        if self._pairs_list is None:
            vid_id = self._vids_list[index % self._num_videos]
            frame_list = self._vids_dict[vid_id]
            vid_a, vid_b = vid_id, vid_id
            frame_a, frame_b = np.random.choice(frame_list, size=2, replace=False)
        else:
            path_a, path_b = self._pairs_list[index % self._num_videos]
            vid_a, frame_a = path_a.split('/')
            vid_b, frame_b = path_b.split('/')

        image_a, mask_a, mano_a = self._get_sample(vid_a, frame_a)
        image_b, mask_b, mano_b = self._get_sample(vid_b, frame_b)

        image_a = self._transform(image_a)
        image_b = self._transform(image_b)

        mask_a = torch.from_numpy(mask_a)
        mask_b = torch.from_numpy(mask_b)

        return {'imageA': image_a, 'maskA': mask_a, 'manoA': mano_a, 'nameA': os.path.join(vid_a, frame_a),
                'imageB': image_b, 'maskB': mask_b, 'manoB': mano_b, 'nameB': os.path.join(vid_b, frame_b)}

    def _get_sample(self, vid_id, frame_id):
        if os.path.exists(os.path.join(self.pic_dir, 'train', vid_id.split('_')[0], 'rgb', frame_id)):
            current_data_split = 'train'
        else:
            current_data_split = 'test'
        image = cv2.imread(os.path.join(self.pic_dir, current_data_split, vid_id.split('_')[0], 'rgb', frame_id))
        mask = cv2.imread(os.path.join(self.pic_dir, current_data_split, vid_id.split('_')[0], 'mask', '%05d.png' % int(frame_id.split('.')[0])))
        mask = cv2.resize(mask, (640, 480))

        bbox = self.bbx_params[vid_id]
        image, trans = augmentation(image, bbox)
        mask, _ = augmentation(mask, bbox)
        trans = trans.astype(np.float32)

        image_inv = (image / 255.0)[:, :, ::-1].copy()
        mask_inv = (mask / 128.0)[None, :, :, -1].copy()

        anno = read_annotation(self.pic_dir, vid_id.split('_')[0], frame_id.split('.')[0], current_data_split)
        objMesh = read_obj(os.path.join('assets/obj', anno['objName'], anno['objName']+'.obj'))

        cam = anno['camMat'].astype(np.float32)
        pose = anno['handPose'].astype(np.float32)
        shape = anno['handBeta'].astype(np.float32)
        handtrans = anno['handTrans'].astype(np.float32)

        vertices_obj = np.zeros((7866, 3), dtype=np.float32)
        vertices_obj_now = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
        vertices_obj[:vertices_obj_now.shape[0]] = vertices_obj_now

        # theta_np = np.concatenate([cam.reshape(-1), trans, pose, shape, vertices_obj])
        # theta = torch.from_numpy(theta_np).float()

        theta = {
            'cam': cam,
            'trans': trans,
            'pose': pose,
            'shape': shape,
            'handtrans': handtrans,
            'vertices_obj': vertices_obj,
            'objName': OBJNAMES.index(anno['objName'])
        }

        return image_inv, mask_inv, theta

    def __len__(self):
        return self._num_videos * self._opt.num_repeats

    def _create_transform(self):
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self._transform = transforms.Compose(transform_list)
