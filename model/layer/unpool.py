import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import coalesce

def save(mesh_name, vert, n_iter, face=None):
    import numpy as np
    import os
    mesh = np.hstack([np.full([vert.shape[0], 1], 'v'), vert.detach().cpu().numpy()])
    if face is not None:
        face = np.hstack([np.full([face.shape[0], 1], 'f'), face.cpu().numpy() + 1])
        mesh = np.vstack([mesh, face])
    out_path = os.path.join('{}_{}.obj'.format(n_iter, mesh_name))
    np.savetxt(out_path, mesh, fmt='%s', delimiter=' ')

class GraphUnpool(nn.Module):
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, x, pool_idx):
        add_feat = torch.sum(x[:, pool_idx], 2) / pool_idx.shape[-1]
        # add_feat = x[:, pool_idx].mean(1)
        outputs = torch.cat([x, add_feat], 1)

        return outputs

class GraphDiffFaceUnpool(nn.Module):
    def __init__(self):
        super(GraphDiffFaceUnpool, self).__init__()

    def forward(self, x, mask, face):
        B = x.shape[0]
        new_vert_list = []
        new_face_list = []
        new_edge_list = []
        for b in range(B):
            pool_face = face[b][mask[b] == 1]
            remain_face = face[b][mask[b] == 0]
            add_feat = torch.sum(x[b, pool_face], dim=1) / pool_face.shape[-1]
            outputs = torch.cat([x[b], add_feat], 0)
            
            V = x[b].shape[0]
            n0 = (torch.arange(add_feat.shape[0]) + V).type_as(pool_face)
            v0, v1, v2 = [pool_face[:, i] for i in range(3)]

            new_face0 = torch.stack([n0, v0, v1], 1)
            new_face1 = torch.stack([n0, v1, v2], 1)
            new_face2 = torch.stack([n0, v2, v0], 1)
            new_faces = torch.cat([remain_face, new_face0, new_face1, new_face2], 0)
            
            # save('test', x[b], 0, face[b])
            # save('test', outputs, 1, new_faces)

            new_edges = torch.cat([new_faces[:, :2], new_faces[:, 1:], new_faces[:, [-1, 0]]], dim=0)
        
            new_vert_list.append(outputs)
            new_face_list.append(new_faces)
            new_edge_list.append(new_edges.t())

        new_verts = torch.stack(new_vert_list, 0)
        new_faces = torch.stack(new_face_list, 0)
        new_edges = torch.stack(new_edge_list, 0)

        return new_verts, new_faces, new_edges

class GraphDiffEdgeUnpool(nn.Module):
    def __init__(self):
        super(GraphDiffEdgeUnpool, self).__init__()

    def forward(self, x, mask, face):
        B = x.shape[0]
        new_vert_list = []
        new_face_list = []
        new_edge_list = []
        for b in range(B):
            pool_face = face[b][mask[b] == 1]

            if pool_face.shape[0] == 0:
                new_edges = torch.cat([face[b][:, :2], face[b][:, 1:], face[b][:, [-1, 0]]], dim=0)
                new_vert_list.append(x[b])
                new_face_list.append(face[b])
                new_edge_list.append(new_edges.t())
                continue

            remain_face = face[b][mask[b] == 0]

            # process chosen face
            pool_edges = [pool_face[:,:2], pool_face[:, 1:], pool_face[:, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
            pool_edges_sort = [e.sort(1)[0] for e in pool_edges]
            
            edge_dict = {}
            start_index = x[b].shape[0]
            
            new_verts = []
            pool_idx = torch.zeros(1, 2).long().to(x.device)
            F1 = pool_face.shape[0]
            for f in range(F1):
                ns = []
                for e in range(3):  
                    edge_key = tuple(pool_edges_sort[e][f].tolist())
                    if edge_key in edge_dict.keys():
                        n = edge_dict[edge_key]
                    else:
                        n = start_index
                        edge_dict[edge_key] = torch.tensor([n]).to(x.device)
                        pool_idx = torch.cat([pool_idx, pool_edges_sort[e][f].unsqueeze(0)], 0)
                        start_index += 1        
                    ns.append(edge_dict[edge_key])
                
                if len(new_verts) == 0:
                    new_verts.extend(ns)
                else:
                    new_verts = [torch.cat([new_verts[i], ns[i]], 0) for i in range(3)]
            
            verts = [pool_face[:, i] for i in range(3)] # [v0 v1 v2]
            v0, v1, v2 = verts
            n0, n1, n2 = new_verts

            new_pool_face0 = torch.stack([v0, n0, n2], 1)
            new_pool_face1 = torch.stack([n0, v1, n1], 1)
            new_pool_face2 = torch.stack([n1, v2, n2], 1)
            new_pool_face3 = torch.stack([n0, n1, n2], 1)
            new_pool_faces = torch.cat([new_pool_face0, new_pool_face1, new_pool_face2, new_pool_face3], 0)

            # add new node
            pool_idx = pool_idx[1:]
            add_feat = torch.sum(x[b, pool_idx], dim=1) / pool_idx.shape[-1]
            outputs = torch.cat([x[b], add_feat], 0)

            if remain_face.shape[0] == 0:
                new_faces = new_pool_faces
                new_edges = torch.cat([new_faces[:, :2], new_faces[:, 1:], new_faces[:, [-1, 0]]], dim=0)

                new_vert_list.append(outputs)
                new_face_list.append(new_faces)
                new_edge_list.append(new_edges.t())
                continue

            # process unchosen face
            # conditions
            # 1 edge
            # [v0, n0, v1, v2] 
            # [v0, n0, v2] [n0, v1, v2] 
            #   0   1  -1    1   2   3
            # 2 edge
            # [v0, n0, v1, n1, v2] 
            # [v0, n0, v2] [n0, v1, n1] [n0, n1, v2] 
            #   0   1  -1    1   2   3    1   3  -1
            # [v0, n0, v1, v2, n2]
            # [v0, n0, n2] [n0, v1, v2] [n0, v2, n2] 
            #   0   1  -1    1   2   3    1   3  -1
            # 3 edge
            # [v0, n0, v1, n1, v2, n2]
            # [v0, n0, n2] [n0, v1, n1] [n0, n1, n2] [n1, v2, n2]
            #   0   1  -1    1   2   3    1   3  -1    3  -2  -1

            remain_edges = [remain_face[:,:2], remain_face[:, 1:], remain_face[:, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
            remain_edges_sort = [e.sort(1)[0] for e in remain_edges]
            
            F2 = remain_face.shape[0]

            index_mask = torch.tensor([[0, 1, -1], [1, 2, 3], [1, 3, -1], [3, -2, -1]]).long()
            
            new_remain_faces_list = []
            for f in range(F2):
                node = []
                first = None
                for e in range(3):
                    edge_key = tuple(remain_edges_sort[e][f].tolist())
                    if edge_key in edge_dict.keys():
                        if first is None:
                            first = e
                            break

                if first is None:
                    new_remain_faces_list.append(remain_face[f])
                    continue
                
                for idx in [-3, -2, -1]:
                    e = idx + first
                    node.append(remain_face[f][e])
                    edge_key = tuple(remain_edges_sort[e][f].tolist())
                    if edge_key in edge_dict.keys():
                        node.append(edge_dict[edge_key].squeeze())
                
                node = torch.stack(node)
                for i in range(len(node)-2):
                    new_face = node[index_mask[i]]
                    new_remain_faces_list.append(new_face)
            
            new_remain_faces = torch.stack(new_remain_faces_list, 0)
            new_faces = torch.cat([new_pool_faces, new_remain_faces], 0)

            # compute edges
            new_edges = torch.cat([new_faces[:, :2], new_faces[:, 1:], new_faces[:, [-1, 0]]], dim=0)

            new_vert_list.append(outputs)
            new_face_list.append(new_faces)
            new_edge_list.append(new_edges.t())

        new_verts = torch.stack(new_vert_list, 0)
        new_faces = torch.stack(new_face_list, 0)
        new_edges = torch.stack(new_edge_list, 0)

        return new_verts, new_faces, new_edges

class GraphDiffLineUnpool(nn.Module):
    def __init__(self):
        super(GraphDiffLineUnpool, self).__init__()

    def forward(self, x, pool_idx, face, mask):
        B = x.shape[0]
        outputs = []
        v_masks = []

        for b in range(B):
            pool_feat = torch.sum(x[b, pool_idx[b]], 1) / pool_idx[b].shape[-1]
            
            mask_feat = mask[b].unsqueeze(-1).expand(-1, pool_feat.shape[-1]) * pool_feat

            ind = mask[b] == 1.
            add_feat = mask_feat[ind]

            pool_vert = pool_idx[b][ind].view(-1)
            v_idx = torch.arange(x.shape[1]).type_as(pool_vert)
            v_mask = torch.any(v_idx[..., None] == pool_vert, dim=-1)
            ones = torch.ones(add_feat.shape[0]).type_as(v_mask)
            v_mask = torch.cat([v_mask, ones], 0)
            v_masks.append(v_mask)
            
            output = torch.cat([x[b], add_feat], 0)
            outputs.append(output)

        outputs = torch.stack(outputs, 0)
        v_masks = torch.stack(v_masks, 0)

        return outputs, v_masks