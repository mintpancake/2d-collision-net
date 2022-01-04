import numpy as np
import torch
import torch.nn as nn
from pytorch_utils import FC, Conv1d, Conv2d
import torch_scatter
from torch.cuda.amp import autocast
from utils import read_config

CFG = read_config()
CUT_SIZE = CFG['cut_size']

SCENE_PT_MLP = [2, 128, 256]
SCENE_VOX_MLP = [256, 512, 1024, 512]
OBJ_MLPS = [2, 64, 128, 256, 512]
CLS_FC = [2050, 1024, 256]


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.bounds = nn.Parameter(
            torch.from_numpy(np.asarray([[-0.5, -0.5], [0.5, 0.5]])).float(), requires_grad=False
        )
        self.vox_size = nn.Parameter(
            torch.from_numpy(np.asarray([0.5 / CUT_SIZE, 0.5 / CUT_SIZE])).float(), requires_grad=False
        )
        self.num_voxels = nn.Parameter(
            ((self.bounds[1] - self.bounds[0]) / self.vox_size).long(),
            requires_grad=False
        )

        self.scene_pt_mlp = nn.Sequential()
        for i in range(len(SCENE_PT_MLP) - 1):
            self.scene_pt_mlp.add_module(
                "pt_layer{}".format(i),
                Conv1d(SCENE_PT_MLP[i], SCENE_PT_MLP[i + 1])
            )

        self.scene_vox_mlp = nn.ModuleList()
        for i in range(len(SCENE_VOX_MLP) - 1):
            scene_conv = nn.Sequential()
            if SCENE_VOX_MLP[i + 1] > SCENE_VOX_MLP[i]:
                scene_conv.add_module(
                    "2d_conv_layer{}".format(i),
                    Conv2d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=3,
                        padding=1
                    ),
                )
                scene_conv.add_module(
                    "2d_max_layer{}".format(i), nn.MaxPool2d(2, stride=2)
                )
            else:
                scene_conv.add_module(
                    "3d_convt_layer{}".format(i),
                    nn.ConvTranspose2d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=2,
                        stride=2
                    ),
                )
            self.scene_vox_mlp.append(scene_conv)

        self.SA = nn.Sequential(
            Conv1d(OBJ_MLPS[0], OBJ_MLPS[1]),
            Conv1d(OBJ_MLPS[1], OBJ_MLPS[2]),
            Conv1d(OBJ_MLPS[2], OBJ_MLPS[3]),
            Conv1d(OBJ_MLPS[3], OBJ_MLPS[4])
        )

        self.obj_FCs = nn.ModuleList(
            [
                FC(OBJ_MLPS[-1], 1024),
                FC(1024, 1024)
            ]
        )

        self.classifier = nn.Sequential(
            FC(CLS_FC[0], CLS_FC[1]),
            FC(CLS_FC[1], CLS_FC[2]),
            FC(CLS_FC[2], 1, activation=None)
        )

    def forward(self, scene_pc, obj_pc, pos):
        scene_features = self.get_scene_features(scene_pc)
        obj_features = self.get_obj_features(obj_pc)
        res = self.classify_tfs(obj_features, scene_features, pos)
        return res

    def get_scene_features(self, scene_pc):
        scene_xy, scene_features = self._break_up_pc(scene_pc)
        scene_inds = self.voxel_inds(scene_xy)
        scene_vox_centers = (
            self._inds_from_flat(scene_inds) * self.vox_size
            + self.vox_size / 2
            + self.bounds[0]
        )
        scene_xy_centered = (scene_pc[..., 0:2] - scene_vox_centers)
        scene_xy_centered.transpose_(2, 1)
        scene_features = self.scene_pt_mlp(scene_xy_centered)
        max_vox_features = torch.zeros(
            (*scene_features.shape[:2], self.num_voxels.prod())
        ).to(scene_pc.device)
        if scene_inds.max() >= self.num_voxels.prod():
            print(
                scene_xy[range(len(scene_pc)), scene_inds.max(axis=-1)[1]],
                scene_inds.max(),
            )
        assert scene_inds.max() < self.num_voxels.prod()
        assert scene_inds.min() >= 0
        with autocast(enabled=False):
            max_vox_features[
                ..., : scene_inds.max() + 1
            ] = torch_scatter.scatter_max(
                scene_features.float(), scene_inds[:, None, :])[0]
        max_vox_features = max_vox_features.reshape(
            *max_vox_features.shape[:2], *self.num_voxels.int()
        )

        l_vox_features = [max_vox_features]
        for i in range(len(self.scene_vox_mlp)):
            li_vox_features = self.scene_vox_mlp[i](l_vox_features[i])
            l_vox_features.append(li_vox_features)

        stack_vox_features = torch.cat(
            (l_vox_features[1], l_vox_features[-1]), dim=1
        )

        stack_vox_features = stack_vox_features.reshape(
            *stack_vox_features.shape[:2], -1
        )

        return stack_vox_features

    def get_obj_features(self, obj_pc):
        obj_xy, obj_features = self._break_up_pc(obj_pc)

        obj_features = self.SA(obj_xy.transpose_(2, 1))
        obj_features = torch.max(obj_features, dim=2).values
        for i in range(len(self.obj_FCs)):
            obj_features = self.obj_FCs[i](obj_features.squeeze(axis=-1))

        return obj_features

    def classify_tfs(self, obj_features, scene_features, trans):
        b = len(scene_features)

        # Get voxel indices for translations
        trans_inds = self.voxel_inds(trans, scale=2).long()
        if trans_inds.max() >= scene_features.shape[2]:
            print(trans[trans_inds.argmax()], trans_inds.max())
        assert trans_inds.max() < scene_features.shape[2]
        assert trans_inds.min() >= 0

        # Calculate translation offsets from centers of voxels
        tr_vox_centers = (
            self._inds_from_flat(trans_inds, scale=2) * self.vox_size * 2
            + self.vox_size / 2
            + self.bounds[0]
        )
        trans_offsets = trans - tr_vox_centers.float()

        # Send concatenated features to classifier
        class_in = torch.cat(
            (
                obj_features.unsqueeze(1).expand(
                    b, scene_features.shape[2], obj_features.shape[-1]),
                scene_features.transpose(2, 1),
                trans_offsets.unsqueeze(1).expand(
                    b, scene_features.shape[2], trans_offsets.shape[-1]),
            ),
            dim=-1,
        )

        return self.classifier(class_in)

    def _break_up_pc(self, pc):
        xy = pc[..., 0:2].contiguous()
        features = None
        return xy, features

    def voxel_inds(self, xy, scale=1):
        inds = torch.div(
            (xy - self.bounds[0]), (scale * self.vox_size), rounding_mode='trunc').int()
        return self._inds_to_flat(inds, scale=scale)

    def _inds_to_flat(self, inds, scale=1):
        flat_inds = inds * torch.cuda.IntTensor(
            [
                torch.div(self.num_voxels[1], scale, rounding_mode='trunc'),
                1,
            ],
            device=self.num_voxels.device,
        )
        return flat_inds.sum(axis=-1)

    def _inds_from_flat(self, flat_inds, scale=1):
        sep = torch.div(self.num_voxels[1], scale, rounding_mode='trunc')
        ind0 = torch.div(flat_inds, sep, rounding_mode='trunc')
        ind1 = flat_inds % sep
        return torch.stack((ind0, ind1), dim=-1)


# model = Net()
# s = torch.Tensor([[[-0.49, -0.49], [-0.24, -0.24], [0.24, 0.24], [0.49, 0.49]], [[0.41, 0.41], [0.31, 0.31],
#                                                                                  [-0.26, -0.26], [0.14, -0.14]], [[-0.31, 0.31], [-0.29, 0.29], [-0.02, 0.02], [0.09, -0.09]]])
# o = torch.Tensor([[[-0.49, -0.49], [-0.24, -0.24], [0.24, 0.24], [0.49, 0.49]], [[0.41, 0.41], [0.31, 0.31],
#                                                                                  [-0.26, -0.26], [0.14, -0.14]], [[-0.31, 0.31], [-0.29, 0.29], [-0.02, 0.02], [0.09, -0.09]]])
# sp = torch.Tensor([[-0.5, -0.5],
#                   [0.09, 0.09],
#                   [0.09, 0.09]])
# res = model(s, o, sp)
# res = res.squeeze()
# res = res.reshape(res.shape[0], 20, 20)
# print(res.shape)
