import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import neural_renderer as nr
from utils import mesh
from utils import util
import os
import cv2
import pickle
from smplx.lbs import transform_mat


def cam2pixel(cam_coord, f, c, trans):
    x = cam_coord[:, :, 0] / (cam_coord[:, :, 2] + 1e-8) * f[:, 0:1] + c[:, 0:1]
    y = cam_coord[:, :, 1] / (cam_coord[:, :, 2] + 1e-8) * f[:, 1:2] + c[:, 1:2]
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


def orthographic_proj_withz_idrot(pts3D, cam, offset_z=0.):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    bs = cam.shape[0]
    cam_mat = cam[:, 0:9].reshape(bs, 3, 3)
    trans = cam[:, 9:].reshape(bs, 2, 3)
    is_OpenGL_coords = True
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 3

    coord_change_mat = torch.from_numpy(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32))[None].cuda()
    coord_change_mat = coord_change_mat.repeat(bs, 1, 1)
    if is_OpenGL_coords:
        pts3D = torch.einsum('ijk,imk->ijm', [pts3D, coord_change_mat])
    proj_pts = torch.einsum('ijk,imk->ijm', [pts3D, cam_mat])
    z = proj_pts[:, :, 2]
    proj_pts = torch.stack([proj_pts[:, :, 0]/proj_pts[:, :, 2], proj_pts[:, :, 1]/proj_pts[:, :, 2]], dim=2)

    assert len(proj_pts.shape) == 3
    proj_xy = torch.cat([proj_pts, torch.ones_like(proj_pts)[:, :, 1:2]], dim=2)
    proj_xy_trans = torch.einsum('ijk,imk->ijm', [trans, proj_xy])
    proj_xy_trans = proj_xy_trans.permute(0, 2, 1)
    proj_xy = torch.cat([proj_xy_trans, z[:, :, None]], dim=2)[:, :, :2]
    proj_xy = proj_xy / 255.0 * 2 - 1
    proj_z = pts3D[:, :, 2:3] + offset_z

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

        obj_root_path = 'assets/obj'
        obj_list = sorted(os.listdir(obj_root_path))

        for j in range(len(obj_list)):
            faces_list = []
            for i in range(2):
                if i == 0:
                    _, cur_faces = nr.load_obj(face_path[0])
                else:
                    obj_uv_path = os.path.join(obj_root_path, obj_list[j], obj_list[j] + '.obj')
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
            obj_uv_path = os.path.join(obj_root_path, obj_list[j], obj_list[j] + '.obj')
            obj_cur_map_fn = torch.tensor(mesh.create_mapping(map_name, obj_uv_path, contain_bg=True,
                                                              fill_back=fill_back)).float()
            sem_tensor_obj = torch.ones(obj_cur_map_fn.shape[0] - 1, 1) * (j + 7)
            sem_tensor_merge = torch.cat([sem_tensor, sem_tensor_obj, torch.zeros(1,1)], dim=0)
            self.register_buffer('sem_full_{}'.format(obj_list[j]), sem_tensor_merge)

        # (nf, T*T, 2)
        hand_cur_map_fn = torch.tensor(mesh.create_mapping(map_name, uv_map_path[0], contain_bg=True,
                                                      fill_back=fill_back)).float()
        for i in range(9):
            map_fn_list = []

            obj_uv_path = os.path.join(obj_root_path, obj_list[i], obj_list[i]+'.obj')
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
        for i in range(9):
            fim_list = []
            wim_list = []
            faces_uv_list = []
            for j in range(2):
                if j == 0:
                    obj_info = mesh.load_obj(face_path[0])
                else:
                    obj_face_path = os.path.join(obj_root_path, obj_list[i], obj_list[i] + '.obj')
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

            object_texture_path = os.path.join(obj_root_path, obj_list[i], 'texture_map.png')
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





if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    render = SMPLRenderer(map_name='uv_seg', fill_back=False,
                          anti_aliasing=True, background_color=(0, 0, 0), has_front=False).cuda()

    smplx_path = ''
    mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True).cuda(),
                  'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True).cuda()}
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


    baseDir = 'HO3D_v3/'
    YCBModelsDir = 'HO3D_YCB_model'
    split = 'train'

    ################################################################################
    # Source img
    seqName = 'SS2'
    id = '0310'
    anno = read_annotation(baseDir, seqName, id, split)
    src_img = read_RGB_img(baseDir, seqName, id, split)
    # objMesh = read_obj(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))
    objMesh = read_obj('assets/object.obj')

    vertices_dict = {}
    root_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[0].view(1, 3).float().cuda()
    hand_pose = torch.from_numpy(anno['handPose']).view(-1, 3)[1:].view(1, -1).float().cuda()
    shape = torch.from_numpy(anno['handBeta']).view(1, -1).float().cuda()
    trans = torch.from_numpy(anno['handTrans']).view(1, 3).float().cuda()
    output = mano_layer['right'](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    vertices = output.vertices
    vertices_dict['hand'] = copy.deepcopy(vertices)
    vertices_obj = torch.from_numpy(np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']).cuda().float()
    vertices_dict['object'] = vertices_obj[None, :]
    cam = torch.from_numpy(anno['camMat']).cuda().float()

    # vertices_numpy = vertices.cpu().numpy()
    # proj_verts = project_3D_points(anno['camMat'], vertices_numpy[0], is_OpenGL_coords=True)
    # for i in range(778):
    #     proj_verts[i][0] = int(proj_verts[i][0])
    #     proj_verts[i][1] = int(proj_verts[i][1])
    # proj_verts = project_3D_points(anno['camMat'], vertices_numpy, is_OpenGL_coords=True)
    # proj_verts_obj = project_3D_points(anno['camMat'], objMesh.v, is_OpenGL_coords=True)
    # plt.figure()
    # plt.imshow(src_img)
    # plt.scatter(proj_verts_obj[:,0], proj_verts_obj[:,1])
    # plt.show()
    # exit()

    bbox = [0, 80.0, 480.0, 480.0]
    src_img, trans = augmentation(src_img, bbox)
    trans = torch.FloatTensor(trans).cuda()
    src_info = {'cam': torch.cat([cam.reshape(-1), trans.reshape(-1)], dim=0)[None, :],
                'verts': torch.cat([vertices_dict['hand'], vertices_dict['object']], dim=1)}
    src_img = torch.from_numpy(src_img).cuda().unsqueeze(0).permute(0, 3, 1, 2)


    ################################################################################
    # Target
    seqName = 'SS2'
    id = '0897'
    tgt_anno = read_annotation(baseDir, seqName, id, split)
    tgt_img = read_RGB_img(baseDir, seqName, id, split)
    objMesh = read_obj('assets/object.obj')


    vertices_dict = {}
    root_pose = torch.from_numpy(tgt_anno['handPose']).view(-1, 3)[0].view(1, 3).float().cuda()
    hand_pose = torch.from_numpy(tgt_anno['handPose']).view(-1, 3)[1:].view(1, -1).float().cuda()
    shape = torch.from_numpy(tgt_anno['handBeta']).view(1, -1).float().cuda()
    trans = torch.from_numpy(tgt_anno['handTrans']).view(1, 3).float().cuda()
    output = mano_layer['right'](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    vertices = output.vertices
    vertices_dict['hand'] = copy.deepcopy(vertices)
    vertices_obj = torch.from_numpy(
        np.matmul(objMesh.v, cv2.Rodrigues(tgt_anno['objRot'])[0].T) + tgt_anno['objTrans']).cuda().float()
    vertices_dict['object'] = vertices_obj[None, :]
    cam = torch.from_numpy(tgt_anno['camMat']).cuda().float()

    bbox = [0, 80.0, 480.0, 480.0]
    tgt_img, trans = augmentation(tgt_img, bbox)
    trans = torch.FloatTensor(trans).cuda()
    ref_info = {'cam': torch.cat([cam.reshape(-1), trans.reshape(-1)], dim=0)[None, :],
                'verts': torch.cat([vertices_dict['hand'], vertices_dict['object']], dim=1)}
    tgt_img = torch.from_numpy(tgt_img).cuda().unsqueeze(0).permute(0, 3, 1, 2)


    ################################################################################
    objname = tgt_anno['objName']
    # Rendering Function
    src_f2verts, src_fim, _ = render.render_fim_wim(src_info['cam'], src_info['verts'], objname)
    src_f2verts = src_f2verts[:, :, :, 0:2]
    src_f2verts[:, :, :, 1] *= -1
    src_cond, _ = render.encode_fim(src_info['cam'], src_info['verts'], objname, fim=src_fim, transpose=True)
    # src_sem, _ = render.encode_sem(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
    src_depth = render.render_depth(src_info['cam'], src_info['verts'], objname)
    src_crop_mask = util.morph(src_cond[:, -1:, :, :], ks=3, mode='erode')

    src_numpy = src_img.cpu().numpy()[0].transpose(1, 2, 0)
    src_cond_numpy = src_cond.cpu().numpy()[0].transpose(1, 2, 0)
    syn_tex = render.get_texture_backward_warp(src_img, src_f2verts, src_fim, objname)
    # render.get_texture_forward_wrap(src_numpy, src_cond_numpy)
    # object_texture_path = 'assets/texture_map.png'
    # obj_tex_img = cv2.imread(object_texture_path)
    # obj_tex_img = cv2.resize(obj_tex_img, (256, 256))
    # obj_tex_img = torch.from_numpy(obj_tex_img)
    # syn_tex[:,:,:,384:] = obj_tex_img.permute(2, 0, 1)[None]

    _, ref_fim, ref_wim = render.render_fim_wim(ref_info['cam'], ref_info['verts'], objname)
    ref_cond, _ = render.encode_fim(ref_info['cam'], ref_info['verts'], objname, fim=ref_fim, transpose=True)
    ref_depth = render.render_depth(ref_info['cam'], ref_info['verts'], objname)
    ref_crop_mask = util.morph(ref_cond[:, -1:, :, :], ks=3, mode='erode')
    T, O = render.cal_bc_transform(src_f2verts, src_fim, ref_fim, ref_wim)
    syn_img = F.grid_sample(src_img, T, align_corners=True)

    O = O.permute(0,3,1,2)
    O = util.morph(O, ks=3, mode='erode')
    O = 1 - util.morph(1 - O, ks=3, mode='erode')
    syn_img = syn_img * (1 - O) + 255 * torch.ones_like(syn_img).cuda() * O

    T_tex = render.sample_from_texture(ref_cond).permute(0, 2, 3, 1)
    syn_img2 = F.grid_sample(syn_tex, T_tex, align_corners=True)

    T_tex_dense = render.sample_from_texture_dense(ref_fim, ref_wim, objname)
    syn_img3 = F.grid_sample(syn_tex, T_tex_dense, align_corners=True)


    ################################################################################
    # Src Products

    plt.figure()
    img_show = src_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0][:,:,::-1].astype(np.uint8)
    plt.imshow(img_show)

    # # semantics mask
    # # COLORS = np.array([[0, 0, 0], [255, 0, 0], [255, 85, 0], [170, 255, 0], [85, 255, 0],
    # #           [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 170, 255],
    # #           [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255]])
    # COLORS = np.array([[0, 0, 0], [214, 48, 49], [255, 118, 117], [225, 112, 85], [250, 177, 160],
    #                    [253, 203, 110], [255, 234, 167], [9, 132, 227], [116, 185, 255],
    #                    [0, 206, 201], [129, 236, 236], [0, 184, 184], [85, 239, 196]])
    #
    # fim_texture = (render.fim_uv).reshape(-1).clone()
    # sem_embed = (render.sem_full).reshape(-1).clone()
    # for i in range(fim_texture.shape[0]):
    #     fim_texture[i] = sem_embed[fim_texture[i]]
    # fim_texture = fim_texture.reshape(256, 640).cpu().numpy()
    # fim_texture = COLORS[fim_texture.reshape(-1)].reshape(256, 640, 3)
    # plt.imsave('sem.png', ((fim_texture).astype(np.uint8)))
    # exit()


    # plt.figure()
    # src_fim_show = src_fim[0].detach().cpu().numpy()
    # plt.imshow(src_fim_show)

    plt.figure(figsize=(25,10))
    plt.imshow((syn_tex[0]).permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.uint8))


    # # img_show = src_cond.permute(0, 2, 3, 1).detach().cpu().numpy()[0,:,:,-1:].astype(np.uint8)
    # plt.figure()
    # img_show = np.array(src_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0][:,:,::-1] *
    #                     (src_crop_mask.detach().cpu().numpy()[0].transpose(1,2,0))).astype(np.uint8)
    # plt.imshow(img_show)
    #
    # # Semantic mask
    # plt.figure()
    # sem_hand = np.array(255.0 / 12 * src_sem[0].detach().cpu().numpy().transpose(1, 2, 0)).astype(
    #     np.uint8)
    # plt.imshow(sem_hand.repeat(3, axis=2))
    #
    #
    #
    #
    # plt.figure()
    # lhand_show = np.array(src_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0][:, :, ::-1] *
    #                       (((src_fim != -1) & (src_fim < 1538)).detach().cpu().numpy().transpose(1, 2, 0))).astype(
    #     np.uint8)
    # # plt.imshow(lhand_show)
    #
    # plt.figure()
    # rhand_show = np.array(src_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0][:, :, ::-1] *
    #                     (((src_fim != -1) & (src_fim >= 1538)).detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8)
    # # plt.imshow(rhand_show)
    #
    # ## Vertices visualization
    # import trimesh
    # bs = src_info['cam'].shape[0]
    # rot = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [1, 0, 0])
    # campos = src_info['cam'][:, 0:3]
    # camrot = src_info['cam'][:, 3:12].reshape(bs, 3, 3)
    # X = src_info['verts'] * 1000.0
    # mesh = world2cam(X.permute(0, 2, 1), camrot, campos[:, :, None]).permute(0, 2, 1)
    # mesh = mesh.cpu().numpy()[0]
    # mesh = mesh.dot(rot[:3,:3])
    # face_path = ['assets/MANO_UV_left.obj', 'assets/MANO_UV_right.obj']
    # faces_list = []
    # for i in range(2):
    #     _, cur_faces = nr.load_obj(face_path[i])
    #     if i == 1:
    #         cur_faces += 778
    #     faces_list.append(cur_faces)
    # faces = torch.cat(faces_list, dim=0).detach().cpu().numpy()
    # save_obj(mesh, faces,
    #          osp.join('save_obj', 'src.obj'))
    #
    # bs = ref_info['cam'].shape[0]
    # rot = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [1, 0, 0])
    # campos = ref_info['cam'][:, 0:3]
    # camrot = ref_info['cam'][:, 3:12].reshape(bs, 3, 3)
    # X = ref_info['verts'] * 1000.0
    # mesh = world2cam(X.permute(0, 2, 1), camrot, campos[:, :, None]).permute(0, 2, 1)
    # mesh = mesh.cpu().numpy()[0]
    # mesh = mesh.dot(rot[:3, :3])
    # face_path = ['assets/MANO_UV_left.obj', 'assets/MANO_UV_right.obj']
    # faces_list = []
    # for i in range(2):
    #     _, cur_faces = nr.load_obj(face_path[i])
    #     if i == 1:
    #         cur_faces += 778
    #     faces_list.append(cur_faces)
    # faces = torch.cat(faces_list, dim=0).detach().cpu().numpy()
    # save_obj(mesh, faces,
    #          osp.join('save_obj', 'ref.obj'))
    #
    # # j2d = orthographic_proj_withz_idrot(src_info['verts'], src_info['cam'])
    # # for i in range(778):
    # #     x, y, _ = j2d[0, i].detach().cpu().numpy()
    # #     x = (x + 1) / 2. * 255.
    # #     y = (y + 1) / 2. * 255.
    # #     plt.plot(x, y, markersize=1.5, marker='.', color='r')
    #
    # # src cond
    # plt.figure()
    # src_cond = flow_vis.flow_to_color(
    #     -1 * src_cond.detach().cpu().numpy()[0].transpose(1, 2, 0)[:, :, :2], convert_to_bgr=False)
    # plt.imshow(1 - src_cond)
    #
    # # ref cond
    # plt.figure()
    # ref_cond = flow_vis.flow_to_color(
    #     -1 * ref_cond.detach().cpu().numpy()[0].transpose(1, 2, 0)[:, :, :2], convert_to_bgr=False)
    # plt.imshow(1-ref_cond)
    #
    # # flow vis
    # plt.figure()
    # flow_color = flow_vis.flow_to_color((T * (T!=-2)).detach().cpu().numpy()[0], convert_to_bgr=False)
    # plt.imshow(flow_color)


    # syn img
    plt.figure()
    plt.imshow(syn_img.detach().cpu().numpy()[0].transpose(1,2,0).astype(np.uint8)[:,:,::-1])

    # # syn img2
    # plt.figure()
    # plt.imshow(syn_img2.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1])

    # syn img3
    plt.figure()
    plt.imshow(syn_img3.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1])

    # hand_region = (1-ref_crop_mask.detach().cpu().numpy()[0].transpose(1,2,0))
    # syn_ref_img = syn_img.detach().cpu().numpy()[0].transpose(1,2,0).astype(np.uint8)[:,:,::-1]
    # blank = np.ones_like(syn_ref_img) * 255
    # output = (syn_ref_img * hand_region + blank * (1 - hand_region)).astype(np.uint8)
    # output_1 = (output == 0) * 255 - (output == 255) * 255 + output
    # plt.imshow(output_1)

    plt.show()
