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


OBJNAMES = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
            '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
            '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
            '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}


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

def read_annotation(base_dir, seq_name, file_id, sample):
    # Add camera.
    fx = sample['intrinsics']['fx']
    fy = sample['intrinsics']['fy']
    cx = sample['intrinsics']['ppx']
    cy = sample['intrinsics']['ppy']
    grasp_id = sample['ycb_grasp_ind']
    cam = torch.Tensor([[fx, fy, cx, cy]]).float()

    betas = torch.tensor(sample['mano_betas'], dtype=torch.float32)

    # grasp_id = 1
    grasp_list = sample['ycb_ids']
    grasp_name = _YCB_CLASSES[grasp_list[grasp_id]]
    label = np.load(os.path.join(base_dir, 'images', seq_name, "labels_{:06d}.npz".format(file_id)))
    pose_y = label['pose_y']
    pose_m = label['pose_m']
    # Add YCB meshes.
    mesh_obj_path = os.path.join(base_dir, 'models', grasp_name, "textured_pre.obj")
    mesh_obj = read_obj(mesh_obj_path)

    pose_obj_list = []
    for o in range(len(pose_y)):
        if np.all(pose_y[o] == 0.0):
            continue
        pose_obj = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
        # pose_obj[1] *= -1
        # pose_obj[2] *= -1
        pose_obj_list.append(pose_obj)

    # Add MANO meshes.
    if not np.all(pose_m == 0.0):
        pose = torch.from_numpy(pose_m)
        cur_mesh = np.array(mesh_obj.v)
        homo_cur_mesh = np.concatenate([cur_mesh, np.ones_like(cur_mesh)[:, 2:]], axis=1)
        homo_transformed = np.matmul(pose_obj_list[grasp_id], homo_cur_mesh.T)
        obj_mesh = homo_transformed[:3].transpose(1, 0)

    return obj_mesh, pose, betas, cam, grasp_name

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


class YCBDataset(DatasetBase):

    def __init__(self, opt, is_for_train=True):
        super(YCBDataset, self).__init__(opt, is_for_train)
        self._name = 'YCBDataset'
        self.data_dir = opt.data_dir
        self.param_dir = os.path.join(opt.data_dir, opt.params_dir)
        self.pic_dir = os.path.join(opt.data_dir, opt.images_dir)
        self.data_split = 'train' if is_for_train else 'test'
        self.pairs_dir = opt.pairs_dir

        if not os.path.exists(self.param_dir):
            raise ValueError("param_dir: %s not exist" % self.param_dir)
        if not os.path.exists(self.pic_dir):
            raise ValueError("pic_dir: %s not exist" % self.pic_dir)

        with open(os.path.join(self.param_dir, 'DexYCB-bbx.pkl'), 'rb') as f:
            self.bbx_params = pickle.load(f)

        with open(os.path.join(self.param_dir, 'valid_video_info.pkl'), 'rb') as f:
            self.cam_params = pickle.load(f)

        _vid_list_dir = os.path.join(self.param_dir, 'DexYCB_train.pkl' if is_for_train else 'DexYCB_test.pkl')
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
            vid_a, frame_a = os.path.join(*path_a.split('/')[:-1]), int(path_a.split('/')[-1])
            vid_b, frame_b = os.path.join(*path_b.split('/')[:-1]), int(path_b.split('/')[-1])

        image_a, mano_a = self._get_sample(vid_a, frame_a)
        image_b, mano_b = self._get_sample(vid_b, frame_b)

        image_a = self._transform(image_a)
        image_b = self._transform(image_b)

        return {'imageA': image_a, 'manoA': mano_a, 'nameA': os.path.join(vid_a, str(frame_a)),
                'imageB': image_b, 'manoB': mano_b, 'nameB': os.path.join(vid_b, str(frame_b))}

    def _get_sample(self, vid_id, frame_id):
        image = cv2.imread(os.path.join(self.pic_dir, vid_id, "color_{:06d}.jpg".format(frame_id)))

        bbox = self.bbx_params[vid_id]
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        image, trans = augmentation(image, bbox)
        trans = trans.astype(np.float32)
        image_inv = (image / 255.0)[:, :, ::-1].copy()

        sample = self.cam_params[vid_id]
        obj_mesh_now, pose, betas, cam, grasp_name = read_annotation(self.data_dir, vid_id, frame_id, sample)
        obj_mesh = np.zeros((8000, 3), dtype=np.float32)
        obj_mesh[:obj_mesh_now.shape[0]] = obj_mesh_now

        # theta_np = np.concatenate([cam.reshape(-1), trans, pose, shape, vertices_obj])
        # theta = torch.from_numpy(theta_np).float()

        theta = {
            'cam': cam[0].float(),
            'trans': trans.astype(np.float32),
            'pose': pose[0].float(),
            'shape': betas.float(),
            'vertices_obj': obj_mesh.astype(np.float32),
            'objName': OBJNAMES.index(grasp_name)
        }

        return image_inv, theta

    def __len__(self):
        return self._num_videos * self._opt.num_repeats

    def _create_transform(self):
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self._transform = transforms.Compose(transform_list)
