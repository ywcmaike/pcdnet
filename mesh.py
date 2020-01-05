import numpy as np
import pickle
import torch
import os
from mesh_utils import *
import time

def save(mesh_name, vert, n_iter, face=None):
    mesh = np.hstack([np.full([vert.shape[0], 1], 'v'), vert.detach().cpu().numpy()])
    if face is not None:
        face = np.hstack([np.full([face.shape[0], 1], 'f'), face.cpu().numpy() + 1])
        mesh = np.vstack([mesh, face])
    out_path = os.path.join('{}_{}.obj'.format(n_iter, mesh_name))
    np.savetxt(out_path, mesh, fmt='%s', delimiter=' ')

class MeshInfo:
    def __init__(self, param, device):

        self.faces = []
        self.unpool_idx = []
        self.lap_idx = []
        self.edges = []

        # level 0
        data0 = load_obj(param.obj_file).to(device)
        self.symm_edge0 = symmetry_edge(data0)
        self.vertices = data0.pos
        self.faces.append(data0.face.t())
        lap_idx0, edge0 = get_adj_list(data0)
        self.edges.append(edge0)
        self.lap_idx.append(lap_idx0)

        save('0', data0.pos, 0, data0.face.t())

        for i in range(param.global_level):
            # level 1
            unpool1, face1 = mesh_edge_subdiv(data0)
            self.unpool_idx.append(unpool1)
            self.faces.append(face1)

            add_vert1 = torch.sum(data0.pos[unpool1], 1) / unpool1.shape[-1]
            new_vert1 = torch.cat([data0.pos, add_vert1], 0)

            data1 = Data(pos=new_vert1, face=face1.t()).to(device)

            save('{}'.format(i+1), data1.pos, 0, data1.face.t())

            lap_idx1, edge1 = get_adj_list(data1)
            self.edges.append(edge1)
            self.lap_idx.append(lap_idx1)
            data0 = data1

        self.sample_points = sample_in_unit_sphere(param.sample_pcl_num)
        save('sample_pcl', self.sample_points, 0)

############# test subdiv #############
        # face = face2
        # V = data2.pos.shape[0]
        # edges = torch.cat([face[:,:2], face[:, 1:], face[:, [-1, 0]]], 0).sort(1)[0]
        # value = edges[:, 0] * V + edges[:, 1]
        # uniq, _ = torch.unique(value, sorted=True, return_inverse=True)
        # edge_node = torch.stack([uniq // V, uniq % V], -1)
        # edge_feats = 0.5 * (data2.pos[edge_node[:, 0]] + data2.pos[edge_node[:, 1]])

        # pool_ind = torch.arange(edge_node.shape[0])
        # pool_edge = edge_node[pool_ind]
        # pool_idx = torch.arange(data2.pos.shape[0], data2.pos.shape[0] + pool_edge.shape[0])

        # FN = face.shape[0]

        # face_edges = [face[:,:2], face[:, 1:], face[:, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
        # edges = torch.cat(face_edges, 0).sort(1)[0]

        # edge_keys = edges[:, 0] * V + edges[:, 1]
        # edge_keys = torch.stack(torch.split(edge_keys, FN), -1)

        # pool_edge_keys = pool_edge[:, 0] * V + pool_edge[:, 1]

        # PE = pool_edge_keys.shape[0]
        # pool_edge_num = torch.zeros(FN).type_as(face)
        # pool_edge_idx = torch.full(face.shape, -1).type_as(face)
        # pool_start_idx = torch.zeros(FN).type_as(face)

        # pool_face_ind_list0 = []
        # pool_face_ind_list1 = []
        # for e in range(PE):
        #     pool_face_mask = torch.eq(pool_edge_keys[e], edge_keys)
        #     pool_edge_idx[pool_face_mask] = pool_idx[e]
        #     pool_face_ind = pool_face_mask.nonzero()

        #     pool_face_ind_list0.append(pool_face_ind[:, 0])
        #     pool_face_ind_list1.append(pool_face_ind[:, 1])

        # pool_face_ind0 = torch.cat(pool_face_ind_list0, 0)
        # pool_face_ind1 = torch.cat(pool_face_ind_list1, 0)

        # pool_edge_num.index_add_(0, pool_face_ind0, torch.ones_like(pool_face_ind0))
        # pool_start_idx.index_add_(0, pool_face_ind0, pool_face_ind1)
            
        # vert_vec = torch.stack([face[:, 0], pool_edge_idx[:, 0], face[:, 1], pool_edge_idx[:, 1], face[:, 2], pool_edge_idx[:,2]], -1)
        # # index_mask = torch.tensor([[0, 1, -1], [1, 2, 3], [1, 3, -1], [3, -2, -1]]).long()

        # # no pool face
        # pool_edge_num_mask0 = (pool_edge_num == 0)
        # new_face0 = face[pool_edge_num_mask0]

        # # 1 pool face
        # pool_edge_num_mask1 = (pool_edge_num == 1)
        # face_vec1 = vert_vec[pool_edge_num_mask1]
        # # roll to align
        # face1_0 = face_vec1[pool_start_idx[pool_edge_num_mask1] == 0]
        # face1_1 = face_vec1[pool_start_idx[pool_edge_num_mask1] == 1].roll(-2, 1)
        # face1_2 = face_vec1[pool_start_idx[pool_edge_num_mask1] == 2].roll(2, 1)
        # face_vec1 = torch.cat([face1_0, face1_1, face1_2], 0)
        # new_face1 = torch.cat(
        #         [torch.stack([face_vec1[:, 0], face_vec1[:, 1], face_vec1[:, 4]], -1), 
        #         torch.stack([face_vec1[:, 1], face_vec1[:, 2], face_vec1[:, 4]], -1)], 0)

        # # 2 pool face
        # pool_edge_num_mask2 = (pool_edge_num == 2)
        # face_vec2 = vert_vec[pool_edge_num_mask2]
        # # roll to align
        # face2_1 = face_vec2[pool_start_idx[pool_edge_num_mask2] == 1]
        # face2_2 = face_vec2[pool_start_idx[pool_edge_num_mask2] == 2].roll(2, 1)
        # face2_3 = face_vec2[pool_start_idx[pool_edge_num_mask2] == 3].roll(-2, 1)
        # face_vec2 = torch.cat([face2_1, face2_2, face2_3], 0)
        # new_face2 = torch.cat(
        #         [torch.stack([face_vec2[:, 0], face_vec2[:, 1], face_vec2[:, 4]], -1), 
        #         torch.stack([face_vec2[:, 1], face_vec2[:, 2], face_vec2[:, 3]], -1),
        #         torch.stack([face_vec2[:, 1], face_vec2[:, 3], face_vec2[:, 4]], -1)], 0)

        # # 3 pool face
        # pool_edge_num_mask3 = (pool_edge_num == 3)
        # face_vec3 = vert_vec[pool_edge_num_mask3]
        # new_face3 = torch.cat(
        #         [torch.stack([face_vec3[:, 0], face_vec3[:, 1], face_vec3[:, 5]], -1), 
        #         torch.stack([face_vec3[:, 1], face_vec3[:, 2], face_vec3[:, 3]], -1),
        #         torch.stack([face_vec3[:, 1], face_vec3[:, 3], face_vec3[:, 5]], -1),
        #         torch.stack([face_vec3[:, 3], face_vec3[:, 4], face_vec3[:, 5]], -1)], 0)

        # new_faces = torch.cat([new_face0, new_face1, new_face2, new_face3], 0)

        # pos = torch.cat([data2.pos, edge_feats[pool_ind]], 0)
        # save('2_sub', pos, 0, new_faces)
        

def create_mesh(param, device=torch.device('cpu')):
    return MeshInfo(param, device)




