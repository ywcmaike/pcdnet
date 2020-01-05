import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from .layer.project import GraphProject, GraphProjectU
from .layer.unpool import GraphUnpool, GraphDiffLineUnpool
from .layer.gsn_conv import GSNConv
from torch_geometric.data import Data, Batch
import sys
sys.path.append('.')
from mesh_utils import *
from loss.chamfer import ChamferDistance
import time

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, num=100):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)[:, 0]
    _, ind = torch.topk(y, num)
    y_hard = torch.zeros_like(y)
    y_hard[ind] = 1.0
    ind = y_hard == 1.0
    return (y_hard - y).detach() + y, ind

class GFaceMask(nn.Module):
    def __init__(self, input_channel, hidden_channel=256):
        super(GFaceMask, self).__init__()
        self.conv1 = GSNConv(input_channel, hidden_channel)
        self.conv2 = GSNConv(hidden_channel, 1)

    def forward(self, input, face):
        B = input.shape[0]
        mask_list = []
        for b in range(B):
            x, edge = dual_mesh(input[b], face[b])
            x = F.relu(self.conv1(x, edge))
            output = torch.tanh(self.conv2(x, edge))
            ones = torch.ones_like(output)
            zeros = torch.zeros_like(output)
            mask = torch.where(output > 0, ones, zeros)
            mask_list.append(mask)

        masks = torch.stack(mask_list, 0).squeeze(-1)
        return masks

class GEdgeMask(nn.Module):
    def __init__(self, input_channel, param, hidden_channel=256):
        super(GEdgeMask, self).__init__()

        self.param = param

        self.conv1 = GSNConv(input_channel, hidden_channel)
        self.conv2 = GSNConv(hidden_channel, 2)

    def forward(self, input, face, coord, detail_points, div=5):
        B = input.shape[0]
        b_mask_lists = []
        b_new_pool_lists = []
        b_new_face_lists = []
        b_new_edge_lists = []

        for b in range(B):

            if not self.param.vote_to_sub:
                x, edge, edge_node = line_mesh(input[b], face[b])
                x, edge, edge_node = line_mesh_slow(input[b], face[b])
                # use line graph gcn
                x = F.relu(self.conv1(x, edge))
                output = torch.sigmoid(self.conv2(x, edge))
                mask, pool_ind = gumbel_softmax(output, 0.1, output.shape[-2] // div)
            else:
                # vote for subdivision
                x, edge, edge_node = line_mesh(coord[b], face[b])
                chamfer_dist = ChamferDistance()
                _, _, vote1, _ = chamfer_dist(detail_points[b].unsqueeze(0), x.unsqueeze(0))
                vote1 = vote1.squeeze(0).detach().long()
                vote_cnt = torch.zeros(x.shape[0]).long().to(face.device)
                vote_cnt.index_add_(0, vote1, torch.ones_like(vote1))
                _, pool_ind = torch.topk(vote_cnt, vote_cnt.shape[0] // div)
                mask = torch.zeros_like(vote_cnt).float()
                mask[pool_ind] = 1.0
                pool_ind = mask == 1.0

            # pool face
            pool_edge = edge_node[pool_ind]
            pool_idx = torch.arange(input[b].shape[0], input[b].shape[0] + pool_edge.shape[0])

            FN = face[b].shape[0]
            V = input[b].shape[0]

            face_edges = [face[b, :,:2], face[b, :, 1:], face[b, :, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
            edges = torch.cat(face_edges, 0).sort(1)[0]

            edge_keys = edges[:, 0] * V + edges[:, 1]
            edge_keys = torch.stack(torch.split(edge_keys, FN), -1)

            pool_edge_keys = pool_edge[:, 0] * V + pool_edge[:, 1]

            PE = pool_edge_keys.shape[0]
            pool_edge_num = torch.zeros(FN).type_as(face)
            pool_edge_idx = torch.full(face[b].shape, -1).type_as(face)
            pool_start_idx = torch.zeros(FN).type_as(face)

            pool_face_ind_list0 = []
            pool_face_ind_list1 = []
            for e in range(PE):
                pool_face_mask = torch.eq(pool_edge_keys[e], edge_keys)
                pool_edge_idx[pool_face_mask] = pool_idx[e]
                pool_face_ind = pool_face_mask.nonzero()

                pool_face_ind_list0.append(pool_face_ind[:, 0])
                pool_face_ind_list1.append(pool_face_ind[:, 1])

            pool_face_ind0 = torch.cat(pool_face_ind_list0, 0)
            pool_face_ind1 = torch.cat(pool_face_ind_list1, 0)

            pool_edge_num.index_add_(0, pool_face_ind0, torch.ones_like(pool_face_ind0))
            pool_start_idx.index_add_(0, pool_face_ind0, pool_face_ind1)
            
            vert_vec = torch.stack([face[b][:, 0], pool_edge_idx[:, 0], face[b][:, 1], pool_edge_idx[:, 1], face[b][:, 2], pool_edge_idx[:,2]], -1)
            # index_mask = torch.tensor([[0, 1, -1], [1, 2, 3], [1, 3, -1], [3, -2, -1]]).long()

            # no pool face
            pool_edge_num_mask0 = (pool_edge_num == 0)
            new_face0 = face[b][pool_edge_num_mask0]

            # 1 pool face
            pool_edge_num_mask1 = (pool_edge_num == 1)
            face_vec1 = vert_vec[pool_edge_num_mask1]
            # roll to align
            face1_0 = face_vec1[pool_start_idx[pool_edge_num_mask1] == 0]
            face1_1 = face_vec1[pool_start_idx[pool_edge_num_mask1] == 1].roll(-2, 1)
            face1_2 = face_vec1[pool_start_idx[pool_edge_num_mask1] == 2].roll(2, 1)
            face_vec1 = torch.cat([face1_0, face1_1, face1_2], 0)
            new_face1 = torch.cat(
                [torch.stack([face_vec1[:, 0], face_vec1[:, 1], face_vec1[:, 4]], -1), 
                torch.stack([face_vec1[:, 1], face_vec1[:, 2], face_vec1[:, 4]], -1)], 0)

            # 2 pool face
            pool_edge_num_mask2 = (pool_edge_num == 2)
            face_vec2 = vert_vec[pool_edge_num_mask2]
            # roll to align
            face2_1 = face_vec2[pool_start_idx[pool_edge_num_mask2] == 1]
            face2_2 = face_vec2[pool_start_idx[pool_edge_num_mask2] == 2].roll(2, 1)
            face2_3 = face_vec2[pool_start_idx[pool_edge_num_mask2] == 3].roll(-2, 1)
            face_vec2 = torch.cat([face2_1, face2_2, face2_3], 0)
            new_face2 = torch.cat(
                [torch.stack([face_vec2[:, 0], face_vec2[:, 1], face_vec2[:, 4]], -1), 
                torch.stack([face_vec2[:, 1], face_vec2[:, 2], face_vec2[:, 3]], -1),
                torch.stack([face_vec2[:, 1], face_vec2[:, 3], face_vec2[:, 4]], -1)], 0)

            # 3 pool face
            pool_edge_num_mask3 = (pool_edge_num == 3)
            face_vec3 = vert_vec[pool_edge_num_mask3]
            new_face3 = torch.cat(
                [torch.stack([face_vec3[:, 0], face_vec3[:, 1], face_vec3[:, 5]], -1), 
                torch.stack([face_vec3[:, 1], face_vec3[:, 2], face_vec3[:, 3]], -1),
                torch.stack([face_vec3[:, 1], face_vec3[:, 3], face_vec3[:, 5]], -1),
                torch.stack([face_vec3[:, 3], face_vec3[:, 4], face_vec3[:, 5]], -1)], 0)

            new_faces = torch.cat([new_face0, new_face1, new_face2, new_face3], 0)
############### 

            # compute edges
            # print(new_face0.shape, new_face1.shape, new_face2.shape, new_face3.shape)
            new_edges = torch.cat([new_faces[:, :2], new_faces[:, 1:], new_faces[:, [-1, 0]]], dim=0)

            all_idx = torch.arange(input[b].shape[0] + pool_edge.shape[0]).type_as(new_edges)
            new_edges = torch.cat([torch.stack([all_idx, all_idx], -1), new_edges], 0)

            b_new_pool_lists.append(edge_node)
            b_new_face_lists.append(new_faces)
            b_new_edge_lists.append(new_edges.t())
            b_mask_lists.append(mask)

        b_new_pool_id = torch.stack(b_new_pool_lists, 0)
        b_new_faces = torch.stack(b_new_face_lists, 0)
        b_new_edges = torch.stack(b_new_edge_lists, 0)
        b_mask = torch.stack(b_mask_lists, 0)

        return b_new_pool_id, b_new_faces, b_new_edges, b_mask
        
class GBottleneck(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GBottleneck, self).__init__()
        self.conv1 = GSNConv(input_channel, output_channel)
        self.conv2 = GSNConv(output_channel, output_channel)

    def forward(self, input, edge, symm_update):
        x = F.relu(self.conv1(input, edge, symm_update))
        x = F.relu(self.conv2(x, edge, symm_update))
        return (input + x) * 0.5

class GResBlock(nn.Module):
    def __init__(self, input_channel, output_channel=3, hidden_channel=256, num_block=6):
        super(GResBlock, self).__init__()

        self.linear = nn.Linear(input_channel, hidden_channel)

        self.conv1 = GCNConv(hidden_channel, hidden_channel)
        self.block2s = nn.ModuleList(
            [GBottleneck(hidden_channel, hidden_channel) for _ in range(num_block)])
        self.conv3 = GCNConv(hidden_channel, output_channel)

    def forward(self, input, edge, symm_update, form_batch):
        xs = []
        outs = []

        # compress feature
        input = self.linear(input)

        if not form_batch:
            for b in range(input.shape[0]):
                symm_edge = None
                if symm_update is not None:
                    symm_edge =  symm_update[b]

                x = F.relu(self.conv1(input[b], edge[b]))
                for i, _ in enumerate(self.block2s):
                    x = self.block2s[i](x, edge[b], symm_edge)
                out = self.conv3(x, edge[b])

                xs.append(x)
                outs.append(out)

        else:
            data_list = []
            for i in range(input.shape[0]):
                data = Data(x=input[i], edge_index=edge[i])
                data_list.append(data)

            batch = Batch.from_data_list(data_list)
            x = F.relu(self.conv1(batch.x, batch.edge_index))
            for i, _ in enumerate(self.block2s):
                x = self.block2s[i](x, batch.edge_index)
            out = self.conv3(x, batch.edge_index)

            batch.x = x
            x_list = batch.to_data_list()
            xs = [d.x for d in x_list]

            batch.x = out
            out_list = batch.to_data_list()
            outs = [o.x for o in out_list]

        return torch.stack(xs, 0), torch.stack(outs, 0)

class GResNet(nn.Module):
    def __init__(self, input_channel, param, output_channel=3):
        super(GResNet, self).__init__()
        self.param = param
        
        self.block1 = GResBlock(input_channel, output_channel, self.param.hidden_channel, self.param.block_num)
        self.unpool = GraphUnpool()
        self.block2_list = nn.ModuleList(
                [GResBlock(input_channel + self.param.hidden_channel, output_channel, self.param.hidden_channel, self.param.block_num) for _ in range(self.param.global_level)])
        self.block3_list = nn.ModuleList(
                [GResBlock(input_channel + self.param.hidden_channel, output_channel, self.param.hidden_channel, self.param.block_num) for _ in range(self.param.increase_level)])

        if param.data_type == 'p2m':
            self.project = GraphProject()

        elif param.data_type == 'shapenet':
            self.project = GraphProjectU(param.use_z_weight)

        if param.use_diff_sub:
            self.gmask_list = nn.ModuleList(
                [GEdgeMask(input_channel + self.param.hidden_channel, param, self.param.hidden_channel) for _ in range(self.param.increase_level)])
            self.diffunpool = GraphDiffLineUnpool()

    def forward(self, img_feats, input, edges, unpool_idxs, proj_mat, faces, symm_edge, detail_points):
        pre_coords = []
        res_coords = []
        res_edges = []
        res_faces = []
        
        # only for subdiv levels
        res_vmasks = []

        B = img_feats.shape[0]
        edge1 = edges[0].unsqueeze(0).repeat(B, 1, 1)
        symm_edge = symm_edge.unsqueeze(0).repeat(B, 1, 1)
        face1 = faces[0].unsqueeze(0).repeat(B, 1, 1)
        symm_update = None
        if self.param.use_symm_edge_update: 
            symm_update = symm_edge

        # inital level
        x = self.project(input, img_feats, proj_mat)
        if self.param.use_symm_edge_aggr:
            edge = torch.cat([edge1, symm_edge], -1)
        else:
            edge = edge1
        
        mid, coord1 = self.block1(x, edge, symm_update, self.param.form_batch)
        if self.param.use_offset:
            coord1 = input + coord1
        
        pre_coords.append(input)
        res_coords.append(coord1)
        res_edges.append(edge1)
        res_faces.append(face1)

        # global levels
        for i in range(self.param.global_level):
            x = self.project(res_coords[-1], img_feats, proj_mat)
            x = torch.cat([x, mid], 2)

            face2 = faces[i + 1].unsqueeze(0).repeat(B, 1, 1)
            coord1_2 = self.unpool(res_coords[-1], unpool_idxs[i])
            x = self.unpool(x, unpool_idxs[i])
            edge2 = edges[i + 1].unsqueeze(0).repeat(B, 1, 1)

            if self.param.use_symm_edge_aggr:
                edge = torch.cat([edge2, symm_edge], -1)
            else:
                edge = edge2

            mid, coord2 = self.block2_list[i](x, edge, symm_update, self.param.form_batch)
            if self.param.use_offset:
                coord2 = coord1_2 + coord2
            
            pre_coords.append(coord1_2)
            res_coords.append(coord2)
            res_edges.append(edge2)
            res_faces.append(face2)

        # increase level
        if self.param.use_diff_sub:
            for i in range(self.param.increase_level):
                x = self.project(res_coords[-1], img_feats, proj_mat)
                x = torch.cat([x, mid], 2)

                unpool_idx2, face3, edge3, mask3 = self.gmask_list[i](x, res_faces[-1], res_coords[-1], detail_points)
                coord2_3, v_masks = self.diffunpool(res_coords[-1], unpool_idx2, face3, mask3)
                x, _= self.diffunpool(x, unpool_idx2, face3, mask3)

                if self.param.use_symm_edge_aggr:
                    edge = torch.cat([edge3, symm_edge], -1)
                else:
                    edge = edge3

                mid, coord3 = self.block3_list[i](x, edge, symm_update, self.param.form_batch)
                if self.param.use_offset:
                    coord3 = coord2_3 + coord3

                pre_coords.append(coord2_3)
                res_coords.append(coord3)
                res_edges.append(edge3)
                res_faces.append(face3)
                res_vmasks.append(v_masks)

        return pre_coords, res_coords, res_edges, res_faces, res_vmasks
