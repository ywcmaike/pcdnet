from .image_model import VGGNet
from .mesh_model import GResNet
from .point_model import ResPointNet
from collections import OrderedDict
from .layer.project import compute_scale_project
import torch.nn as nn
import torch
from .pcdnet import PointCloudDeformNet

class Model(nn.Module):
    def __init__(self, init_mesh, param):
        super(Model, self).__init__()
        
        self.vertices = nn.Parameter(init_mesh.vertices, False)
        self.symm_edge0 = nn.Parameter(init_mesh.symm_edge0, False)
        self.edges = nn.ParameterList([nn.Parameter(init_mesh.edges[i], False) for i in range(len(init_mesh.edges))])
        self.unpool_idxs = nn.ParameterList([nn.Parameter(init_mesh.unpool_idx[i], False) for i in range(len(init_mesh.unpool_idx))])
        self.faces = nn.ParameterList([nn.Parameter(init_mesh.faces[i], False) for i in range(len(init_mesh.faces))])
        self.sample_points = nn.Parameter(init_mesh.sample_points, False)

        self.param = param
        self.image_model = VGGNet() 
        self.mesh_model = GResNet(input_channel=1011, param=param)
        self.point_model = ResPointNet(input_channel=1011, param=param)
        self.point_deform_model = PointCloudDeformNet(param=param)

    def forward(self, img, proj_mat, logger, n_iter):
        img_featses = self.image_model(img)
        img_feats = img_featses[0]
        # print("img_feats: ", img_feats.shape)

        B = img.shape[0]
        input = self.vertices.unsqueeze(0).repeat(B, 1, 1)
        points = self.sample_points.unsqueeze(0).repeat(B, 1, 1)

        if self.param.use_pcdnet:
            output_points = self.point_deform_model(img_featses, points, proj_mat)
            # print("pcdnet output: ", output_points.shape)
        else:
            output_points = self.point_model(img_feats, points, proj_mat)
        pre_coords, coords, edges, faces, vmasks = self.mesh_model(img_feats, input, self.edges, self.unpool_idxs, proj_mat, self.faces, self.symm_edge0, output_points)
        output_img = img_feats[-1]
        return output_points, pre_coords, coords, edges, faces, vmasks, output_img

def create_model(init_mesh, param, train, checkpoint=None):
    model = Model(init_mesh, param)
    if checkpoint is not None:
        model_dict = torch.load(checkpoint)
        model.load_state_dict(model_dict)
        print("load checkpoint file ", checkpoint)
    if train:
        model = model.train()
    else:
        model = model.eval()
    return model
