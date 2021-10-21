import torch
import torch.nn as nn
from pointnet3.arch.yanx27_pointnet_util import PointNetSetAbstraction
from pytorch_utils import FC, Conv1d, Conv2d


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            Conv1d(2, 128),
            Conv1d(128, 256)
        )
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        ))
        self.conv.append(nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        ))
        self.conv.append(nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, 2)
        ))
        self.SA = nn.Sequential(
            Conv1d(2, 64),
            Conv1d(64, 128),
            Conv1d(128, 256),
            Conv1d(256, 512)
            # PointNetSetAbstraction(npoint=256, radius=0.02,
            #                        nsample=64, mlp=[0, 64, 128]),
            # PointNetSetAbstraction(npoint=64, radius=0.04,
            #                        nsample=128, mlp=[128, 128, 256]),
            # PointNetSetAbstraction(npoint=None, radius=None,
            #                        nsample=None, mlp=[256, 256, 512])
        )
        self.obj_FCs = nn.ModuleList([FC(512, 1024), FC(1024, 1024)])
        self.classifier = nn.Sequential(
            FC(1538, 1024),
            FC(1024, 256),
            FC(256, 1, activation=None),
        )

    def forward(self, sc, oc, pos, device):
        num = len(sc)
        cut_size = 20
        cut = torch.linspace(-0.5, 0.5, cut_size+1, dtype=torch.float32).to(device)
        voxel_features = torch.zeros([num, 256, cut_size, cut_size]).to(device)
        for n in range(num):
            for i in range(len(cut)-1):
                for j in range(len(cut)-1):
                    sc_voxel = sc[n, (sc[n, :, 0] > cut[i]) & (sc[n, :, 0] <= cut[i+1]) &
                                  (sc[n, :, 1] > cut[j]) & (sc[n, :, 1] <= cut[j+1])]
                    if(len(sc_voxel) == 0):
                        continue
                    center = torch.Tensor(
                        [(cut[i]+cut[i+1])/2, (cut[j]+cut[j+1])/2]).to(device)
                    sc_voxel_centered = sc_voxel-center
                    
                    mlp_in = torch.zeros([1, 2, len(sc_voxel_centered)]).to(device)
                    mlp_in[0, :, :] = torch.transpose(sc_voxel_centered, 0, 1)
                    sc_voxel_feature = self.mlp(mlp_in)
                    max_pool_feature = torch.reshape(torch.max(
                        sc_voxel_feature, dim=2).values, [256, ])
                    voxel_features[n, :, i, j] = max_pool_feature
        l_vox_features = [voxel_features]
        for i in range(len(self.conv)):
            li_vox_features = self.conv[i](l_vox_features[i])
            l_vox_features.append(li_vox_features)
        scene_features = torch.cat(
            (l_vox_features[1], l_vox_features[-1]), dim=1
        )
        # scene_features = scene_features.reshape(
        #     *scene_features.shape[:2], -1
        # )

        obj_center = torch.mean(oc, axis=1, keepdim=True)
        oc_centered = oc-obj_center
        SA_in = torch.zeros([num, 2, len(oc_centered)]).to(device)
        SA_in = torch.transpose(oc_centered, 1, 2)
        obj_features = self.SA(SA_in)
        obj_features = torch.reshape(torch.max(
            obj_features, dim=2).values, [num, 512, 1])
        # xy, obj_features = self.SA(oc, None)
        # for i in range(len(self.obj_FCs)):
        #     obj_features = self.obj_FCs[i](obj_features.squeeze(axis=-1))

        new_cut_size = scene_features.shape[2]
        new_cut = torch.linspace(-0.5, 0.5,
                                 new_cut_size+1, dtype=torch.float32).to(device)

        scene_features = torch.reshape(
            scene_features, (scene_features.shape[0], scene_features.shape[1], scene_features.shape[2]*scene_features.shape[3]))

        obj_features = torch.cat(scene_features.shape[2]*[obj_features], dim=2)
        class_in = torch.cat([scene_features, obj_features], dim=1)
        rel_pos = torch.zeros([num, 2, 10, 10]).to(device)
        for i in range(new_cut_size):
            for j in range(new_cut_size):
                center = torch.Tensor(
                    [(new_cut[i]+new_cut[i+1])/2, (new_cut[j]+new_cut[j+1])/2]).to(device)
                rel_pos[:, :, i, j] = pos-center
        rel_pos = torch.reshape(
            rel_pos, (rel_pos.shape[0], rel_pos.shape[1], rel_pos.shape[2]*rel_pos.shape[3]))
        class_in = torch.cat([class_in, rel_pos], dim=1)
        class_in = torch.transpose(class_in, 1, 2)
        return self.classifier(class_in)
