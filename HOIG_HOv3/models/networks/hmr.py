import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import h5py
from .batch_mano import MANO
import os
import smplx
from smplx.lbs import transform_mat


def proj_func(points, cam):
    transl, rot, cen = cam
    bs = points.shape[0]
    with torch.no_grad():
        camera_mat = torch.zeros([bs, 2, 2]).cuda()
        camera_mat[:, 0, 0] = 5000.0
        camera_mat[:, 1, 1] = 5000.0

    rotation = torch.eye(3).unsqueeze(dim=0).repeat(bs, 1, 1).cuda()
    translation = torch.zeros([bs, 3]).cuda()
    center = torch.zeros([bs, 2]).cuda()
    rotation[:] = rot
    translation[:] = transl
    center[:] = cen

    camera_transform = transform_mat(rotation, translation.unsqueeze(dim=-1))
    homog_coord = torch.ones(list(points.shape)[:-1] + [1]).cuda()
    # Convert the points to homogeneous coordinates
    points_h = torch.cat([points, homog_coord], dim=-1)

    projected_points = torch.einsum('bki,bji->bjk',
                                    [camera_transform, points_h])

    img_points = torch.div(projected_points[:, :, :2],
                           projected_points[:, :, 2].unsqueeze(dim=-1))
    img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
                 + center.unsqueeze(dim=1)
    img_points = img_points / 255.0 * 2 - 1

    return img_points


class HandModelRecovery(nn.Module):
    """
        regressor can predict betas(include beta and theta which needed by MANO) from coder
        extracted from encoder in a iteration way
    """

    def __init__(self, mano_path, feature_dim=2048, theta_dim=31):
        super(HandModelRecovery, self).__init__()

        # define mano
        self.mano_layer_right = smplx.create(mano_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True)

        # self.mano_layer['left'] = nn.DataParallel(self.mano_layer['left']).cuda()
        # self.mano_layer['right'] = nn.DataParallel(self.mano_layer['right']).cuda()

        self.feature_dim = feature_dim
        self.theta_dim = theta_dim

    def get_details(self, theta):
        """
            purpose:
                calc verts, joint2d, joint3d, Rotation matrix

            inputs:
                theta: N X (3 + 72 + 10)

            return:
                thetas, verts, j2d, j3d, Rs
        """
        bs = theta['cam'].shape[0]
        cam = theta['cam']
        trans = theta['trans']
        handtrans = theta['handtrans']
        root_pose = theta['pose'][:, :3]
        hand_pose = theta['pose'][:, 3:]
        shape = theta['shape']
        output = self.mano_layer_right(global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=handtrans)
        vertices_hand = output.vertices
        vertices_obj = theta['vertices_obj']

        detail_info = {
            'cam': torch.cat([cam.reshape(bs, -1), trans.reshape(bs, -1)], dim=1),
            'verts': torch.cat([vertices_hand, vertices_obj], dim=1),
            'objName': theta['objName']
        }

        return detail_info
