import torch
import torch.nn as nn
import pickle
import numpy as np


class MANO(nn.Module):
    def __init__(self, pose_num, pkl_path):
        super(MANO, self).__init__()
        self.bases_num = 10
        self.pose_num = pose_num
        self.mesh_num = 778
        self.keypoints_num = 16

        self.dd = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
        self.kintree_table = self.dd['kintree_table']
        self.id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        self.parent = {i: self.id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}

        self.register_buffer("mesh_mu", torch.from_numpy(np.expand_dims(self.dd['v_template'], 0).astype(np.float32)))
        self.register_buffer("mesh_pca", torch.from_numpy(np.expand_dims(self.dd['shapedirs'], 0).astype(np.float32)))
        self.register_buffer("posedirs", torch.from_numpy(np.expand_dims(self.dd['posedirs'], 0).astype(np.float32)))
        self.register_buffer("J_regressor", torch.from_numpy(
            np.expand_dims(self.dd['J_regressor'].todense(), 0).astype(np.float32)))
        self.register_buffer("weights", torch.from_numpy(np.expand_dims(self.dd['weights'], 0).astype(np.float32)))
        self.register_buffer("hands_components", torch.from_numpy(
            np.expand_dims(np.vstack(self.dd['hands_components'][:self.pose_num]), 0).astype(np.float32)))
        self.register_buffer("hands_mean", torch.from_numpy(np.expand_dims(self.dd['hands_mean'], 0).astype(np.float32)))
        self.register_buffer("root_rot", torch.Tensor([np.pi, 0., 0.]).unsqueeze(0))


    def rodrigues(self, r):
        theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1))

        def S(n_):
            ns = torch.split(n_, 1, 1)
            Sn_ = torch.cat([torch.zeros_like(ns[0]), -ns[2], ns[1], ns[2],
                             torch.zeros_like(ns[0]), -ns[0], -ns[1], ns[0], torch.zeros_like(ns[0])], 1)
            Sn_ = Sn_.view(-1, 3, 3)
            return Sn_

        n = r / (theta.view(-1, 1))
        Sn = S(n).to(theta)
        I3 = torch.eye(3).unsqueeze(0).to(theta)
        R = I3 + torch.sin(theta).view(-1, 1, 1) * Sn + (1. - torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn, Sn)
        Sr = S(r)
        theta2 = theta ** 2
        R2 = I3 + (1. - theta2.view(-1, 1, 1) / 6.) * Sr + (.5 - theta2.view(-1, 1, 1) / 24.) * torch.matmul(Sr, Sr)
        idx = np.argwhere((theta < 1e-30).data.cpu().numpy())

        if (idx.size):
            R[idx, :, :] = R2[idx, :, :]

        return R, Sn

    def get_poseweights(self, poses, bsize):
        # pose: batch x 24 x 3
        pose_matrix, _ = self.rodrigues(poses[:, 1:, :].contiguous().view(-1, 3))
        pose_matrix = pose_matrix - torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),
                                                               bsize * (self.keypoints_num - 1), axis=0)).to(pose_matrix)
        pose_matrix = pose_matrix.view(bsize, -1)
        return pose_matrix

    def rot_pose_beta_to_mesh(self, rots, poses, betas):
        batch_size = rots.size(0)
        poses = (self.hands_mean + torch.matmul(poses.unsqueeze(1),
                                                self.hands_components).squeeze(1)).view(batch_size,
                                                                                        self.keypoints_num - 1, 3)
        poses = torch.cat((self.root_rot.repeat(batch_size, 1).view(batch_size, 1, 3), poses), 1)

        v_shaped = (torch.matmul(betas.unsqueeze(1),
                                 self.mesh_pca.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2).contiguous().view(
                                     batch_size, self.bases_num, -1)).squeeze(1)
                    + self.mesh_mu.repeat(batch_size, 1, 1).view(batch_size, -1)).view(batch_size, self.mesh_num, 3)

        pose_weights = self.get_poseweights(poses, batch_size)

        v_posed = v_shaped + torch.matmul(self.posedirs.repeat(batch_size, 1, 1, 1), (pose_weights.view(batch_size, 1, (self.keypoints_num - 1) * 9, 1)).repeat(1, self.mesh_num, 1, 1)).squeeze(3)

        J_posed = torch.matmul(v_shaped.permute(0, 2, 1), self.J_regressor.repeat(batch_size, 1, 1).permute(0, 2, 1))
        J_posed = J_posed.permute(0, 2, 1)
        J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]

        pose = poses.permute(1, 0, 2)
        pose_split = torch.split(pose, 1, 0)

        angle_matrix = []
        for i in range(self.keypoints_num):
            out, tmp = self.rodrigues(pose_split[i].contiguous().view(-1, 3))
            angle_matrix.append(out)

        with_zeros = lambda x: \
            torch.cat((x, torch.Tensor([[[0.0, 0.0, 0.0, 1.0]]]).cuda().repeat(batch_size, 1, 1)), 1).cuda()
        pack = lambda x: torch.cat((torch.zeros(batch_size, 4, 3).cuda(), x), 2)

        results = {}
        results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size, 3, 1)), 2))

        for i in range(1, self.kintree_table.shape[1]):
            tmp = with_zeros(torch.cat((angle_matrix[i],
                                        (J_posed_split[i] - J_posed_split[self.parent[i]]).view(batch_size, 3, 1)), 2))
            results[i] = torch.matmul(results[self.parent[i]], tmp)
        results_global = results

        results2 = []

        for i in range(len(results)):
            vec = (torch.cat((J_posed_split[i], torch.zeros(batch_size, 1).cuda()), 1)).view(batch_size, 4, 1).cuda()
            results2.append((results[i] - pack(torch.matmul(results[i], vec))).unsqueeze(0))

        results = torch.cat(results2, 0)

        T = torch.matmul(results.permute(1, 2, 3, 0),
                         self.weights.repeat(batch_size, 1, 1).permute(0, 2, 1).unsqueeze(1).repeat(1, 4, 1, 1))
        Ts = torch.split(T, 1, 2)
        rest_shape_h = torch.cat((v_posed, torch.ones(batch_size, self.mesh_num, 1).cuda()), 2)
        rest_shape_hs = torch.split(rest_shape_h, 1, 2)

        v = Ts[0].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1,
                                                                                                       self.mesh_num) \
            + Ts[1].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1,
                                                                                                         self.mesh_num) \
            + Ts[2].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1,
                                                                                                         self.mesh_num) \
            + Ts[3].contiguous().view(batch_size, 4, self.mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1,
                                                                                                         self.mesh_num)

        Rots = self.rodrigues(rots)[0]

        Jtr = []

        for j_id in range(len(results_global)):
            Jtr.append(results_global[j_id][:, :3, 3:4])

        # Add finger tips from mesh to joint list
        Jtr.insert(4, v[:, :3, 333].unsqueeze(2))
        Jtr.insert(8, v[:, :3, 444].unsqueeze(2))
        Jtr.insert(12, v[:, :3, 672].unsqueeze(2))
        Jtr.insert(16, v[:, :3, 555].unsqueeze(2))
        Jtr.insert(20, v[:, :3, 745].unsqueeze(2))

        Jtr = torch.cat(Jtr, 2)  # .permute(0,2,1)

        v = torch.matmul(Rots, v[:, :3, :]).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)
        Jtr = torch.matmul(Rots, Jtr).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)

        return torch.cat((Jtr, v), 1)


    def forward(self, theta, beta, cam):
        rot = theta[:, 0:3]
        pose = theta[:, 3:]
        shape = beta
        x3d = self.rot_pose_beta_to_mesh(rot, pose, shape)

        scale = cam[:, 0]
        trans = cam[:, 1:3]
        x2d = trans.unsqueeze(1) + scale.unsqueeze(1).unsqueeze(2) * x3d[:, :, :2]

        return x3d[:, 21:], x3d[:, 0:21], x2d[:, :21]