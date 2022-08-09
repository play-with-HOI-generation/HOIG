import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import neural_renderer as nr
from utils import mesh
from utils import util
import os
import cv2
# import flow_vis
import pickle
import copy

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

def cam2pixel(cam_coord, f, c, trans):
  x = cam_coord[:, :, 0] / (cam_coord[:, :, 2] + 1e-8) * f[:, 0] + c[:, 0]
  y = cam_coord[:, :, 1] / (cam_coord[:, :, 2] + 1e-8) * f[:, 1] + c[:, 1]
  z = cam_coord[:, :, 2]

  proj_xy = torch.cat([x[:, None], y[:, None], torch.ones_like(x)[:, None]], dim=1)
  # return proj_xy
  proj_xy_trans = torch.einsum('ijk,ikm->ijm', [trans, proj_xy])
  # proj_xy_trans = proj_xy
  img_coord = torch.cat([proj_xy_trans, z[:, None]], dim=1)
  return img_coord.permute(0, 2, 1)


def cam2pixel_bak(cam_coord, f, c, trans):
    x = cam_coord[:, :, 0] / (cam_coord[:, :, 2] + 1e-8) * f[:, 0] + c[:, 0]
    y = cam_coord[:, :, 1] / (cam_coord[:, :, 2] + 1e-8) * f[:, 1] + c[:, 1]
    z = cam_coord[:, :, 2]

    proj_xy = torch.cat([x[:,None], y[:,None], torch.ones_like(x)[:,None]], dim=1)
    proj_xy_trans = torch.einsum('ijk,ikm->ijm', [trans, proj_xy])
    # proj_xy_trans = proj_xy
    img_coord = torch.cat([proj_xy_trans, z[:,None]], dim=1)

    return img_coord.permute(0, 2, 1)

def world2cam(world_coord, R, T):
    cam_coord = torch.einsum('ijk,ikm->ijm', [R, (world_coord-T)])
    return cam_coord

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


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


def orthographic_proj_withz_idrot(X, cam, offset_z=0.):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    bs = cam.shape[0]
    focal = cam[:, 0:2]
    princpt = cam[:, 2:4]
    trans = cam[:, 4:].reshape(bs, 2, 3)
    proj_xy = cam2pixel(X, focal, princpt, trans)[:, :, :2]
    proj_xy = proj_xy / 255.0 * 2 - 1
    # proj_z = X[:, :, 2:3] / 1000.0 + offset_z
    proj_z = X[:, :, 2:3] + offset_z

    return torch.cat((proj_xy, proj_z), dim=2)



def orthographic_proj_withz_idrot_bak(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 3: [sc, tx, ty]
    No rotation!
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    bs = cam.shape[0]
    campos = cam[:, 0:3]
    camrot = cam[:, 3:12].reshape(bs, 3, 3)
    focal = cam[:, 12:14]
    princpt = cam[:, 14:16]
    trans = cam[:, 16:22].reshape(bs, 2, 3)

    X = X * 1000.0
    X_cam = world2cam(X.permute(0, 2, 1), camrot, campos[:, :, None]).permute(0, 2, 1)
    X_cam = X_cam / 1000.0

    proj_xy = cam2pixel(X_cam, focal, princpt, trans)[:, :, :2]
    proj_xy = proj_xy / 255.0 * 2 - 1

    proj_z = X[:, :, 2:3] / 1000.0 + offset_z

    return torch.cat((proj_xy, proj_z), dim=2)


def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X_rot
    proj = X_rot

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x

    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


class MANORenderer(nn.Module):
    def __init__(self, face_path=['assets/MANO_UV_right.obj'],
                 uv_map_path=['assets/MANO_UV_right.obj'],
                 map_name='uv_seg', tex_size=3, image_size=256,
                 anti_aliasing=True, fill_back=False, background_color=(0, 0, 0), viewing_angle=30, near=0.1, far=25.0,
                 has_front=False):
        """
        Args:
            face_path:
            uv_map_path:
            map_name:
            tex_size:
            image_size:
            anti_aliasing:
            fill_back:
            background_color:
            viewing_angle:
            near:
            far:
            has_front:
        """

        super(MANORenderer, self).__init__()

        self.background_color = background_color
        self.anti_aliasing = anti_aliasing
        self.image_size = image_size
        self.fill_back = fill_back
        self.map_name = map_name

        self.tex_size = tex_size
        self.register_buffer('coords', self.create_coords(tex_size))


        # Load YCB meshes.
        obj_root_path = '/mnt/blob/data/DexYCB_sub/models/'
        obj_list = []
        for k, v in _YCB_CLASSES.items():
            obj_list.append(v)


        for j in range(len(obj_list)):
            faces_list = []
            for i in range(2):
                if i == 0:
                    _, cur_faces = nr.load_obj(face_path[0])
                else:
                    obj_uv_path = os.path.join(obj_root_path, obj_list[j], "textured_pre.obj")
                    _, cur_faces = nr.load_obj(obj_uv_path)
                    cur_faces += 778
                faces_list.append(cur_faces)
            faces = torch.cat(faces_list, dim=0)
            # self.base_nf = faces.shape[0]
            # fill back
            if self.fill_back:
                faces = np.concatenate((faces, faces[:, ::-1]), axis=0)
            faces = faces.int()
            # self.nf = faces.shape[0]
            self.register_buffer('faces_{}'.format(obj_list[j]), faces)

        with open('assets/semantics_hand.pkl', 'rb') as f:
            sem_hand = pickle.load(f)
        sem_list = []

        for j in range(len(obj_list)):
            sem_tensor = torch.zeros(1538, 1)
            for i, key in enumerate(['palm', 'thumb', 'index_finger', 'middle_finger', 'ring_finger', 'little_finger']):
                sem_tensor[sem_hand['right'][key]] = i + 1
            # sem_list.append(sem_tensor.clone())
            # sem_list.append(torch.zeros(1,1))
            # sem_full = torch.cat(sem_list, dim=0)
            obj_uv_path = os.path.join(obj_root_path, obj_list[j], "textured_pre.obj")
            obj_cur_map_fn = torch.tensor(mesh.create_mapping(map_name, obj_uv_path, contain_bg=True,
                                                              fill_back=fill_back)).float()
            sem_tensor_obj = torch.ones(obj_cur_map_fn.shape[0] - 1, 1) * (j + 7)
            sem_tensor_merge = torch.cat([sem_tensor, sem_tensor_obj, torch.zeros(1,1)], dim=0)
            self.register_buffer('sem_full_{}'.format(obj_list[j]), sem_tensor_merge)

        # (nf, T*T, 2)
        hand_cur_map_fn = torch.tensor(mesh.create_mapping(map_name, uv_map_path[0], contain_bg=True,
                                                      fill_back=fill_back)).float()
        for i in range(len(obj_list)):
            map_fn_list = []

            obj_uv_path = os.path.join(obj_root_path, obj_list[i], "textured_pre.obj")
            obj_cur_map_fn = torch.tensor(mesh.create_mapping(map_name, obj_uv_path, contain_bg=True,
                                                      fill_back=fill_back)).float()

            obj_cur_map_fn[:-1, :2] = obj_cur_map_fn[:-1, :2] + torch.tensor([1.5, 0.0]) * (i + 1)

            map_fn_list.append(hand_cur_map_fn[:-1])
            map_fn_list.append(obj_cur_map_fn)
            map_fn = torch.cat(map_fn_list, dim=0)
            self.register_buffer('map_fn_{}'.format(obj_list[i]), map_fn)

        # back_map_fn = torch.tensor(mesh.create_mapping('back', uv_map_path, contain_bg=True,
        #                                                fill_back=fill_back)).float()
        # self.register_buffer('back_map_fn', back_map_fn)

        if has_front:
            front_map_fn = torch.tensor(mesh.create_mapping('front', uv_map_path, contain_bg=True,
                                                            fill_back=fill_back)).float()
            self.register_buffer('front_map_fn', front_map_fn)
        else:
            self.front_map_fn = None

        # light
        self.light_intensity_ambient = 1
        self.light_intensity_directional = 0
        self.light_color_ambient = [1, 1, 1]
        self.light_color_directional = [1, 1, 1]
        self.light_direction = [0, 1, 0]

        self.rasterizer_eps = 1e-3

        # project function and camera
        self.near = near
        self.far = far
        self.proj_func = orthographic_proj_withz_idrot
        self.viewing_angle = viewing_angle
        self.eye = [0, 0, -(1. / np.tan(np.radians(self.viewing_angle)) + 1)]

        # # uv_map
        for i in range(len(obj_list)):
            fim_list = []
            wim_list = []
            faces_uv_list = []
            for j in range(2):
                if j == 0:
                    obj_info = mesh.load_obj(face_path[0])
                else:
                    obj_face_path = os.path.join(obj_root_path, obj_list[i], "textured_pre.obj")
                    obj_info = mesh.load_obj(obj_face_path)
                vts_ori = obj_info['vts']
                # vts[:, 1] = 1 - vts[:, 1]
                vts_ori = torch.from_numpy(vts_ori)[None]
                vts = (vts_ori - 0.5) * 2
                uv_vert = torch.cat([vts, torch.ones_like(vts[:, :, 0:1])], dim=2)
                uv_vert = nr.look_at(uv_vert, self.eye)
                faces_uv = torch.LongTensor(obj_info['faces_vts'])[None]
                if j == 0:
                    uv_vert_new = (uv_vert + 1) / 2
                    faces_uv_list.append(uv_vert_new[0, faces_uv])
                else:
                    uv_vert_new = (uv_vert + 1) / 2
                    faces_uv_list.append(uv_vert_new[0, faces_uv]+torch.Tensor([1.5, 0, 0])[None, None, None])
                # rasterization
                faces_uv = nr.vertices_to_faces(uv_vert, faces_uv).cuda()

                fim, wim = nr.rasterize_face_index_map_and_weight_map(faces_uv, self.image_size, False)
                fim_list.append(fim)
                wim_list.append(wim)
            empty_fim = -1 * torch.ones(1, 256, 128).int()
            empty_wim = torch.zeros(1, 256, 128, 3)
            fim_uv = torch.cat([fim_list[0], empty_fim.cuda(), fim_list[1]+((fim_list[1]!=-1) * 1538)], dim=2)
            wim_uv = torch.cat([wim_list[0], empty_wim.cuda(), wim_list[1]], dim=2)
            faces_uv_coord = torch.cat(faces_uv_list, dim=1)[:,:,:,:2]
            mean = torch.Tensor([[1.25, 0.5]])
            scale = torch.Tensor([[0.8, -2]])
            faces_uv_coord = (faces_uv_coord - mean) * scale

            self.register_buffer('fim_uv_{}'.format(obj_list[i]), fim_uv)
            self.register_buffer('wim_uv_{}'.format(obj_list[i]), wim_uv)
            self.register_buffer('faces_uv_coord_{}'.format(obj_list[i]), faces_uv_coord)

            object_texture_path = os.path.join(obj_root_path, obj_list[i], 'texture_map_resize.png')
            obj_tex_img = cv2.imread(object_texture_path)[:,:,::-1]
            obj_tex_img = cv2.resize(obj_tex_img, (256, 256))
            obj_tex_img = torch.from_numpy(obj_tex_img.astype(np.float32)) / 255.0 * 2.0 - 1
            self.register_buffer('obj_tex_img_{}'.format(obj_list[i]), obj_tex_img)

    def set_ambient_light(self, int_dir=0.3, int_amb=0.7, direction=(1, 0.5, 1)):
        self.light_intensity_directional = int_dir
        self.light_intensity_ambient = int_amb
        if direction is not None:
            self.light_direction = direction

    def set_bgcolor(self, color=(-1, -1, -1)):
        self.background_color = color

    def set_tex_size(self, tex_size):
        del self.coords
        self.coords = self.create_coords(tex_size).cuda()

    def forward(self, cam, vertices, uv_imgs, obj_name, dynamic=True, get_fim=False):
        bs = cam.shape[0]
        # faces = self.faces.repeat(bs, 1, 1)
        faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)

        if dynamic:
            samplers = self.dynamic_sampler(cam, vertices, faces)
        else:
            samplers = self.img2uv_sampler.repeat(bs, 1, 1, 1)

        textures = self.extract_tex(uv_imgs, samplers)

        images, fim = self.render(cam, vertices, textures, faces, get_fim=get_fim)

        if get_fim:
            return images, textures, fim
        else:
            return images, textures

    def render(self, cam, vertices, textures, obj_name, faces=None, get_fim=False):
        if faces is None:
            bs = cam.shape[0]
            # faces = self.faces.repeat(bs, 1, 1)
            faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)

        # lighting is inplace operation
        textures = textures.clone()
        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(faces, textures, self.image_size, self.anti_aliasing,
                              self.near, self.far, self.rasterizer_eps, self.background_color)
        fim = None
        if get_fim:
            fim = nr.rasterize_face_index_map(faces, image_size=self.image_size, anti_aliasing=False,
                                              near=self.near, far=self.far, eps=self.rasterizer_eps)

        return images, fim

    def render_fim(self, cam, vertices, obj_name, faces=None):
        if faces is None:
            bs = cam.shape[0]
            # faces = self.faces.repeat(bs, 1, 1)
            faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)


        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim = nr.rasterize_face_index_map(faces, self.image_size, False)
        return fim

    def render_fim_wim(self, cam, vertices, obj_name, faces=None):
        if faces is None:
            bs = cam.shape[0]
            # faces = self.faces.repeat(bs, 1, 1)
            faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)


        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim, wim = nr.rasterize_face_index_map_and_weight_map(faces, self.image_size, False)
        return faces, fim, wim

    def render_depth(self, cam, vertices, obj_name):
        bs = cam.shape[0]
        # faces = self.faces.repeat(bs, 1, 1)
        faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_silhouettes(self, cam, vertices, obj_name, faces=None):
        if faces is None:
            bs = cam.shape[0]
            # faces = self.faces.repeat(bs, 1, 1)
            faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)


        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def infer_face_index_map(self, cam, vertices):
        raise NotImplementedError
        # bs = cam.shape[0]
        # faces = self.faces.repeat(bs, 1, 1)
        #
        # # if self.fill_back:
        # #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
        #
        # vertices = self.weak_projection(cam, vertices)
        #
        # # rasterization
        # faces = nr.vertices_to_faces(vertices, faces)
        # fim = nr.rasterize_face_index_map(faces, self.image_size, False)

        # return fim

    def encode_fim(self, cam, vertices, obj_name, fim=None, transpose=True, map_fn=None):

        if fim is None:
            fim = self.infer_face_index_map(cam, vertices)

        if map_fn is not None:
            fim_enc = map_fn[fim.long()]
        else:
            # fim_enc = self.map_fn[fim.long()]
            fim_enc = getattr(self, 'map_fn_{}'.format(obj_name))[fim.long()]

        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc, fim

    def encode_sem(self, cam, vertices, obj_name, fim=None, transpose=True, sem_full=None):

        if fim is None:
            fim = self.infer_face_index_map(cam, vertices)

        if sem_full is None:
            sem_full = getattr(self, 'sem_full_{}'.format(obj_name))
        sem_enc = sem_full[fim.long()]

        if transpose:
            sem_enc = sem_enc.permute(0, 3, 1, 2)

        return sem_enc, fim

    def encode_front_fim(self, fim, transpose=True, front_fn=True):
        if front_fn:
            fim_enc = self.front_map_fn[fim.long()]
        else:
            fim_enc = self.back_map_fn[fim.long()]

        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc

    def extract_tex_from_image(self, images, cam, vertices, obj_name):
        bs = images.shape[0]
        # faces = self.faces.repeat(bs, 1, 1)
        faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)

        sampler = self.dynamic_sampler(cam, vertices, faces)  # (bs, nf, T*T, 2)

        tex = self.extract_tex(images, sampler)

        return tex

    def extract_tex(self, uv_img, uv_sampler):
        """
        :param uv_img: (bs, 3, h, w)
        :param uv_sampler: (bs, nf, T*T, 2)
        :return:
        """

        # (bs, 3, nf, T*T)
        tex = F.grid_sample(uv_img, uv_sampler, align_corners=True)
        # (bs, 3, nf, T, T)
        tex = tex.view(-1, 3, self.nf, self.tex_size, self.tex_size)
        # (bs, nf, T, T, 3)
        tex = tex.permute(0, 2, 3, 4, 1)
        # (bs, nf, T, T, T, 3)
        tex = tex.unsqueeze(4).repeat(1, 1, 1, 1, self.tex_size, 1)

        return tex

    def dynamic_sampler(self, cam, vertices, faces):
        # ipdb.set_trace()
        points = self.batch_orth_proj_idrot(cam, vertices)  # (bs, nf, 2)
        faces_points = self.points_to_faces(points, faces)   # (bs, nf, 3, 2)
        # print(faces_points.shape)
        sampler = self.points_to_sampler(self.coords, faces_points)  # (bs, nf, T*T, 2)
        return sampler

    def project_to_image(self, cam, vertices):
        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        # proj_verts[:, :, 1] *= -1
        proj_verts = proj_verts[:, :, 0:2]
        return proj_verts

    def points_to_faces(self, points, obj_name, faces=None):
        """
        :param points:
        :param faces
        :return:
        """
        bs, nv = points.shape[:2]
        device = points.device

        if faces is None:
            # faces = self.faces.repeat(bs, 1, 1)
            faces = getattr(self, 'faces_{}'.format(obj_name)).repeat(bs, 1, 1)

            # if self.fill_back:
            #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        points = points.reshape((bs * nv, 2))
        # pytorch only supports long and byte tensors for indexing
        return points[faces.long()]

    @staticmethod
    def compute_barycenter(f2vts):
        """

        :param f2vts:  N x F x 3 x 2
        :return: N x F x 2
        """

        # Compute alpha, beta (this is the same order as NMR)
        v2 = f2vts[:, :, 2]  # (nf, 2)
        v0v2 = f2vts[:, :, 0] - f2vts[:, :, 2]  # (nf, 2)
        v1v2 = f2vts[:, :, 1] - f2vts[:, :, 2]  # (nf, 2)

        fbc = v2 + 0.5 * v0v2 + 0.5 * v1v2

        return fbc

    @staticmethod
    def batch_orth_proj_idrot(camera, X):
        """
        X is N x num_points x 3
        camera is N x 3
        same as applying orth_proj_idrot to each N
        """

        # TODO check X dim size.
        # X_trans is (N, num_points, 2)
        X_trans = X[:, :, :2] + camera[:, None, 1:]
        # reshape X_trans, (N, num_points * 2)
        # --- * operation, (N, 1) x (N, num_points * 2) -> (N, num_points * 2)
        # ------- reshape, (N, num_points, 2)

        return camera[:, None, 0:1] * X_trans

    @staticmethod
    def points_to_sampler(coords, faces):
        """
        :param coords: [2, T*T]
        :param faces: [batch size, number of vertices, 3, 2]
        :return: [batch_size, number of vertices, T*T, 2]
        """

        # Compute alpha, beta (this is the same order as NMR)
        nf = faces.shape[1]
        v2 = faces[:, :, 2]  # (bs, nf, 2)
        v0v2 = faces[:, :, 0] - faces[:, :, 2]  # (bs, nf, 2)
        v1v2 = faces[:, :, 1] - faces[:, :, 2]  # (bs, nf, 2)

        # bs x  F x 2 x T*2
        samples = torch.matmul(torch.stack((v0v2, v1v2), dim=-1), coords) + v2.view(-1, nf, 2, 1)
        # bs x F x T*2 x 2 points on the sphere
        samples = samples.permute(0, 1, 3, 2)
        samples = torch.clamp(samples, min=-1.0, max=1.0)
        return samples

    @staticmethod
    def create_coords(tex_size=3):
        """
        :param tex_size: int
        :return: 2 x (tex_size * tex_size)
        """
        if tex_size == 1:
            step = 1
        else:
            step = 1 / (tex_size - 1)

        alpha_beta = torch.arange(0, 1+step, step, dtype=torch.float32).cuda()
        xv, yv = torch.meshgrid([alpha_beta, alpha_beta])

        coords = torch.stack([xv.flatten(), yv.flatten()], dim=0)

        return coords

    @staticmethod
    def create_meshgrid(image_size):
        """
        Args:
            image_size:

        Returns:
            (image_size, image_size, 2)
        """
        factor = torch.arange(0, image_size, dtype=torch.float32) / (image_size - 1)   # [0, 1]
        factor = (factor - 0.5) * 2
        xv, yv = torch.meshgrid([factor, factor])
        # grid = torch.stack([xv, yv], dim=-1)
        grid = torch.stack([yv, xv], dim=-1)
        return grid

    @staticmethod
    def get_vis_f2pts(f2pts, fims):
        """
        Args:
            f2pts: (bs, f, 3, 2) or (bs, f, 3, 3)
            fims:  (bs, 256, 256)

        Returns:

        """

        def get_vis(orig_f2pts, fim):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                fim: (256, 256)

            Returns:
                vis_f2pts: (f, 3, 2)
            """
            vis_f2pts = torch.zeros_like(orig_f2pts) - 2.0
            # 0 is -1
            face_ids = fim.unique()[1:].long()
            vis_f2pts[face_ids] = orig_f2pts[face_ids]

            return vis_f2pts

        # import ipdb
        # ipdb.set_trace()
        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_vis(f2pts[i], fims[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_vis(f2pts, fims)

        return all_vis_f2pts

    @staticmethod
    def set_null_f2pts(f2pts, fims):
        """
        Args:
            f2pts: (bs, f, 3, 2) or (bs, f, 3, 3)
            fims:  (bs, 256, 256)

        Returns:

        """

        def get_vis(orig_f2pts, fim):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                fim: (256, 256)

            Returns:
                vis_f2pts: (f, 3, 2)
            """
            # 0 is -1
            face_ids = fim.unique()[1:].long()
            orig_f2pts[face_ids] = -2.0

            return orig_f2pts

        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_vis(f2pts[i], fims[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_vis(f2pts, fims)

        return all_vis_f2pts

    def cal_transform(self, bc_f2pts, src_fim, dst_fim):
        """
        Args:
            bc_f2pts:
            src_fim:
            dst_fim:

        Returns:

        """
        device = bc_f2pts.device
        bs = src_fim.shape[0]
        # T = renderer.init_T.repeat(bs, 1, 1, 1)    # (bs, image_size, image_size, 2)
        T = (torch.zeros(bs, self.image_size, self.image_size, 2, device=device) - 2)
        # 2. calculate occlusion flows, (bs, no, 2)
        dst_ids = dst_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)

        for i in range(bs):
            Ti = T[i]

            tgt_i = dst_ids[i]

            # (nf, 2)
            tgt_flows = bc_f2pts[i, dst_fim[i, tgt_i].long()]  # (nt, 2)
            Ti[tgt_i] = tgt_flows

        return T

    def cal_bc_transform(self, src_f2pts, src_fims, dst_fims, dst_wims):
        """
        Args:
            src_f2pts: (bs, 13776, 3, 2)
            dst_fims:  (bs, 256, 256)
            dst_wims:  (bs, 256, 256, 3)
        Returns:

        """
        bs = src_f2pts.shape[0]
        T = -2 * torch.ones((bs, self.image_size * self.image_size, 2), dtype=torch.float32, device=src_f2pts.device)
        O = torch.zeros((bs, self.image_size * self.image_size, 1), dtype=torch.float32, device=src_f2pts.device)


        for i in range(bs):
            # (13776, 3, 2)
            from_faces_verts_on_img = src_f2pts[i]

            # to_face_index_map
            to_face_index_map = dst_fims[i]

            # src_vis_list = list(set(list(src_fims[0].reshape(-1).cpu().numpy())))
            # src_vis_list.pop(-1)

            # to_weight_map
            to_weight_map = dst_wims[i]

            # from_face_index_map
            from_face_index_map = src_fims[i]

            # (256, 256) -> (256*256, )
            to_face_index_map = to_face_index_map.long().reshape(-1)
            # (256, 256, 3) -> (256*256, 3)
            to_weight_map = to_weight_map.reshape(-1, 3)
            # (256, 256) -> (256*256, )
            from_face_index_map = from_face_index_map.long().reshape(-1)

            to_exist_mask = (to_face_index_map != -1)
            # to_visible_mask = torch.stack([(to_face_index_map == i) for i in src_vis_list]).sum(0)
            # occlusion_mask = to_exist_mask.int() - to_visible_mask

            # (exist_face_num,)
            to_exist_face_idx = to_face_index_map[to_exist_mask]
            # (exist_face_num, 3)
            to_exist_face_weights = to_weight_map[to_exist_mask]

            # (exist_face_num, 3, 2) * (exist_face_num, 3) -> sum -> (exist_face_num, 2)
            exist_smpl_T = (from_faces_verts_on_img[to_exist_face_idx] * to_exist_face_weights[:, :, None]).sum(dim=1)
            # (256, 256, 2)
            T[i, to_exist_mask] = exist_smpl_T
            # O[i, :, 0] = occlusion_mask

            # exist_T_ld = ((exist_smpl_T + 1) / 2.0 * 255.0).long()
            # exist_T_rd = exist_T_ld.clone() + torch.tensor([1, 0], dtype=exist_T_ld.dtype).cuda()
            # exist_T_lu = exist_T_ld.clone() + torch.tensor([0, 1], dtype=exist_T_ld.dtype).cuda()
            # exist_T_ru = exist_T_ld.clone() + torch.tensor([1, 1], dtype=exist_T_ld.dtype).cuda()

            exist_T_11 = ((exist_smpl_T.clamp(-1., 1.) + 1) / 2.0 * 255.0).long()
            exist_T_00 = (exist_T_11.clone() + torch.tensor([-1, -1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_01 = (exist_T_11.clone() + torch.tensor([-1, 0], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_02 = (exist_T_11.clone() + torch.tensor([-1, 1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_10 = (exist_T_11.clone() + torch.tensor([0, -1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_12 = (exist_T_11.clone() + torch.tensor([0, 1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_20 = (exist_T_11.clone() + torch.tensor([1, -1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_21 = (exist_T_11.clone() + torch.tensor([1, 0], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_22 = (exist_T_11.clone() + torch.tensor([1, 1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)

            # exist_O_ld = from_face_index_map[exist_T_ld[:, 1] * 256 + exist_T_ld[:, 0]]
            # exist_O_rd = from_face_index_map[exist_T_rd[:, 1] * 256 + exist_T_rd[:, 0]]
            # exist_O_lu = from_face_index_map[exist_T_lu[:, 1] * 256 + exist_T_lu[:, 0]]
            # exist_O_ru = from_face_index_map[exist_T_ru[:, 1] * 256 + exist_T_ru[:, 0]]
            # to_visible_mask = (exist_O_ld == to_exist_face_idx) | (exist_O_rd == to_exist_face_idx) | \
            #                   (exist_O_lu == to_exist_face_idx) | (exist_O_ru == to_exist_face_idx)

            exist_O_00 = from_face_index_map[exist_T_00[:, 1] * 256 + exist_T_00[:, 0]]
            exist_O_01 = from_face_index_map[exist_T_01[:, 1] * 256 + exist_T_01[:, 0]]
            exist_O_02 = from_face_index_map[exist_T_02[:, 1] * 256 + exist_T_02[:, 0]]
            exist_O_10 = from_face_index_map[exist_T_10[:, 1] * 256 + exist_T_10[:, 0]]
            exist_O_11 = from_face_index_map[exist_T_11[:, 1] * 256 + exist_T_11[:, 0]]
            exist_O_12 = from_face_index_map[exist_T_12[:, 1] * 256 + exist_T_12[:, 0]]
            exist_O_20 = from_face_index_map[exist_T_20[:, 1] * 256 + exist_T_20[:, 0]]
            exist_O_21 = from_face_index_map[exist_T_21[:, 1] * 256 + exist_T_21[:, 0]]
            exist_O_22 = from_face_index_map[exist_T_22[:, 1] * 256 + exist_T_22[:, 0]]
            to_visible_mask = (exist_O_00 == to_exist_face_idx) | (exist_O_01 == to_exist_face_idx) | \
                              (exist_O_02 == to_exist_face_idx) | (exist_O_10 == to_exist_face_idx) | \
                              (exist_O_11 == to_exist_face_idx) | (exist_O_12 == to_exist_face_idx) | \
                              (exist_O_20 == to_exist_face_idx) | (exist_O_21 == to_exist_face_idx) | \
                              (exist_O_22 == to_exist_face_idx)

            O[i, to_exist_mask, 0] = 1 - to_visible_mask.float()

        T = T.view(bs, self.image_size, self.image_size, 2)
        O = O.view(bs, self.image_size, self.image_size, 1)

        return T, O

    def debug_textures(self):
        return torch.ones((self.nf, self.tex_size, self.tex_size, self.tex_size, 3), dtype=torch.float32)

    def get_texture_backward_warp(self, im, src_f2verts, src_fims, obj_name, pre_load=True):
        bs = src_f2verts.shape[0]
        T = -2 * torch.ones((bs, 256 * 640, 2), dtype=torch.float32, device=src_f2verts.device)
        O = torch.zeros((bs, 256 * 640, 1), dtype=torch.float32, device=src_f2verts.device)

        for i in range(bs):
            # (13776, 3, 2)
            from_faces_verts_on_img = src_f2verts[i]

            # to_face_index_map
            # to_face_index_map = self.fim_uv[0]
            to_face_index_map = getattr(self, 'fim_uv_{}'.format(obj_name))[0]


            # to_weight_map
            # to_weight_map = self.wim_uv[0]
            to_weight_map = getattr(self, 'wim_uv_{}'.format(obj_name))[0]

            # (256, 640) -> (256*640, )
            to_face_index_map = to_face_index_map.long().reshape(-1)
            # (256, 640, 3) -> (256*640, 3)
            to_weight_map = to_weight_map.reshape(-1, 3)

            to_exist_mask = (to_face_index_map != -1)
            # (exist_face_num,)
            to_exist_face_idx = to_face_index_map[to_exist_mask]
            # (exist_face_num, 3)
            to_exist_face_weights = to_weight_map[to_exist_mask]

            # (exist_face_num, 3, 2) * (exist_face_num, 3) -> sum -> (exist_face_num, 2)
            exist_smpl_T = (from_faces_verts_on_img[to_exist_face_idx] * to_exist_face_weights[:, :, None]).sum(dim=1)
            # (256, 640, 2)
            T[i, to_exist_mask] = exist_smpl_T

            # from_face_index_map
            from_face_index_map = src_fims[i]
            from_face_index_map = from_face_index_map.long().reshape(-1)


            exist_T_11 = ((exist_smpl_T + 1) / 2.0 * 255.0).long().clamp(0, 255)
            exist_T_00 = (exist_T_11.clone() + torch.tensor([-1, -1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_01 = (exist_T_11.clone() + torch.tensor([-1, 0], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_02 = (exist_T_11.clone() + torch.tensor([-1, 1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_10 = (exist_T_11.clone() + torch.tensor([0, -1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_12 = (exist_T_11.clone() + torch.tensor([0, 1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_20 = (exist_T_11.clone() + torch.tensor([1, -1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_21 = (exist_T_11.clone() + torch.tensor([1, 0], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)
            exist_T_22 = (exist_T_11.clone() + torch.tensor([1, 1], dtype=exist_T_11.dtype).cuda()).clamp(0, 255)

            # exist_O_ld = from_face_index_map[exist_T_ld[:, 1] * 256 + exist_T_ld[:, 0]]
            # exist_O_rd = from_face_index_map[exist_T_rd[:, 1] * 256 + exist_T_rd[:, 0]]
            # exist_O_lu = from_face_index_map[exist_T_lu[:, 1] * 256 + exist_T_lu[:, 0]]
            # exist_O_ru = from_face_index_map[exist_T_ru[:, 1] * 256 + exist_T_ru[:, 0]]
            # to_visible_mask = (exist_O_ld == to_exist_face_idx) | (exist_O_rd == to_exist_face_idx) | \
            #                   (exist_O_lu == to_exist_face_idx) | (exist_O_ru == to_exist_face_idx)

            exist_O_00 = from_face_index_map[exist_T_00[:, 1] * 256 + exist_T_00[:, 0]]
            exist_O_01 = from_face_index_map[exist_T_01[:, 1] * 256 + exist_T_01[:, 0]]
            exist_O_02 = from_face_index_map[exist_T_02[:, 1] * 256 + exist_T_02[:, 0]]
            exist_O_10 = from_face_index_map[exist_T_10[:, 1] * 256 + exist_T_10[:, 0]]
            exist_O_11 = from_face_index_map[exist_T_11[:, 1] * 256 + exist_T_11[:, 0]]
            exist_O_12 = from_face_index_map[exist_T_12[:, 1] * 256 + exist_T_12[:, 0]]
            exist_O_20 = from_face_index_map[exist_T_20[:, 1] * 256 + exist_T_20[:, 0]]
            exist_O_21 = from_face_index_map[exist_T_21[:, 1] * 256 + exist_T_21[:, 0]]
            exist_O_22 = from_face_index_map[exist_T_22[:, 1] * 256 + exist_T_22[:, 0]]
            to_visible_mask = (exist_O_00 == to_exist_face_idx) | (exist_O_01 == to_exist_face_idx) | \
                              (exist_O_02 == to_exist_face_idx) | (exist_O_10 == to_exist_face_idx) | \
                              (exist_O_11 == to_exist_face_idx) | (exist_O_12 == to_exist_face_idx) | \
                              (exist_O_20 == to_exist_face_idx) | (exist_O_21 == to_exist_face_idx) | \
                              (exist_O_22 == to_exist_face_idx)

            O[i, to_exist_mask, 0] = 1 - to_visible_mask.float()

        T = T.view(bs, 256, 640, 2)
        O = O.view(bs, 256, 640, 1)
        syn_tex = F.grid_sample(im, T)

        O = O.permute(0, 3, 1, 2)
        O = util.morph(O, ks=3, mode='erode')
        O = 1 - util.morph(1 - O, ks=3, mode='erode')
        syn_tex = syn_tex * (1 - O) + 1.0 * torch.ones_like(syn_tex).cuda() * O

        if pre_load:
            syn_tex[:, :, :, 384:] = getattr(self, 'obj_tex_img_{}'.format(obj_name)).permute(2, 0, 1)[None]

        return syn_tex

    def sample_from_texture(self, cond):
        uv = cond[:, :2]
        bs = cond.shape[0]
        mean = torch.Tensor([[1.25, 0.5]])[:,:,None,None].cuda().repeat(bs, 1, 1, 1)
        scale = torch.Tensor([[0.8, 2]])[:,:,None,None].cuda().repeat(bs, 1, 1, 1)
        T_tex = (uv != 0) * (uv - mean) * scale + (uv == 0) * torch.ones_like(uv).cuda() * (-2)
        return T_tex

    def sample_from_texture_dense(self, fim, wim, obj_name):
        bs = fim.shape[0]
        T = -2 * torch.ones((bs, 256 * 256, 2), dtype=torch.float32, device=fim.device)

        for i in range(bs):
            # (13776, 3, 2)
            # from_faces_verts_on_img = self.faces_uv_coord[0]
            from_faces_verts_on_img = getattr(self, 'faces_uv_coord_{}'.format(obj_name))[0]

            # to_face_index_map
            to_face_index_map = fim[i]

            # to_weight_map
            to_weight_map = wim[i]

            # (256, 256) -> (256*256, )
            to_face_index_map = to_face_index_map.long().reshape(-1)
            # (256, 256, 3) -> (256*256, 3)
            to_weight_map = to_weight_map.reshape(-1, 3)

            to_exist_mask = (to_face_index_map != -1)
            # (exist_face_num,)
            to_exist_face_idx = to_face_index_map[to_exist_mask]
            # (exist_face_num, 3)
            to_exist_face_weights = to_weight_map[to_exist_mask]

            # (exist_face_num, 3, 2) * (exist_face_num, 3) -> sum -> (exist_face_num, 2)
            exist_smpl_T = (from_faces_verts_on_img[to_exist_face_idx] * to_exist_face_weights[:, :, None]).sum(dim=1)
            # (256, 256, 2)
            T[i, to_exist_mask] = exist_smpl_T
        T = T.view(bs, 256, 256, 2)

        return T



def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


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

def read_RGB_img(base_dir, seq_name, file_id, split):
    """Read the RGB image in dataset"""
    if os.path.exists(os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')):
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')
    else:
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.jpg')

    _assert_exist(img_filename)

    img = cv2.imread(img_filename)

    return img

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

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts
