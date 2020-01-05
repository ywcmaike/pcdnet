from loss import *
from mesh_utils import *
import os

def save(mesh_name, vert, face=None):
    mesh = np.hstack([np.full([vert.shape[0], 1], 'v'), vert.detach().cpu().numpy()])
    if face is not None:
        face = np.hstack([np.full([face.shape[0], 1], 'f'), face.cpu().numpy() + 1])
        mesh = np.vstack([mesh, face])
    out_path = os.path.join('{}.obj'.format(mesh_name))
    np.savetxt(out_path, mesh, fmt='%s', delimiter=' ')

if __name__ == '__main__':
    pos = torch.tensor([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]).float()
    face = torch.tensor([[0, 1, 2], [2, 3, 0], [6, 5, 4], [4, 7, 6], 
                        [5, 1, 0], [0, 4, 5], [3, 2, 6], [6, 7, 3], 
                        [4, 0, 3], [3, 7, 4], [2, 1, 5], [5, 6, 2]])
    save('test', pos, face)

    sample_points, sample_normals = reparam_sample(pos, face, 5)
    print(sample_points)
    print(sample_normals)

    query_points = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]]).float()
    query_normals = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]).float()

    sample_points = sample_points.unsqueeze(0)
    sample_normals = sample_normals.unsqueeze(0)

    query_points = query_points.unsqueeze(0)
    query_normals = query_normals.unsqueeze(0)

    print(compute_orient_chamfer_loss(sample_points, query_points, sample_normals, query_normals))
    
     