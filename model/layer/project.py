import torch
import torch.nn as nn

class GraphProject(nn.Module):
    def __init__(self):
        super(GraphProject, self).__init__()
        self.intrinsic = [250, 250, 112, 112]
        self.img_size = [224, 224]
    
    def forward(self, vertices, img_feats, proj_mat):
        return self.project(vertices, img_feats)

    def project(self, point3d, img_feats):
        fx, fy, cx, cy = self.intrinsic
        w0, h0 = self.img_size
        
        X = point3d[:, :, 0]
        Y = point3d[:, :, 1]
        Z = point3d[:, :, 2]

        h = fy * torch.div(Y, Z) + cy
        w = fx * torch.div(X, -Z) + cx

        x = h/(1.0 * h0/56)
        y = w/(1.0 * w0/56)
        x = torch.clamp(x, 0, 55)
        y = torch.clamp(y, 0, 55)
        out1 = torch.stack([self.bilinear_interpolate(img_feats[0][i], x[i], y[i]) for i in range(point3d.shape[0])], 0)

        x = h/(1.0 * h0/28)
        y = w/(1.0 * w0/28)
        x = torch.clamp(x, 0, 27)
        y = torch.clamp(y, 0, 27)
        out2 = torch.stack([self.bilinear_interpolate(img_feats[1][i], x[i], y[i]) for i in range(point3d.shape[0])], 0)

        x = h/(1.0 * h0/14)
        y = w/(1.0 * w0/14)
        x = torch.clamp(x, 0, 13)
        y = torch.clamp(y, 0, 13)
        out3 = torch.stack([self.bilinear_interpolate(img_feats[2][i], x[i], y[i]) for i in range(point3d.shape[0])], 0)

        x = h/(1.0 * h0/7)
        y = w/(1.0 * w0/7)
        x = torch.clamp(x, 0, 6)
        y = torch.clamp(y, 0, 6)
        out4 = torch.stack([self.bilinear_interpolate(img_feats[3][i], x[i], y[i]) for i in range(point3d.shape[0])], 0)
        outputs = torch.cat([point3d, out1, out2, out3,out4], 2)
        return outputs

    def bilinear_interpolate(self, img_feat, x, y):
        x1 = torch.floor(x).long()
        x2 = torch.ceil(x).long()
        y1 = torch.floor(y).long()
        y2 = torch.ceil(y).long()

        Q11 = img_feat[:, x1, y1]
        Q12 = img_feat[:, x1, y2]
        Q21 = img_feat[:, x2, y1]
        Q22 = img_feat[:, x2, y2]

        x1 = x1.float()
        x2 = x2.float()
        y1 = y1.float()
        y2 = y2.float()

        weights = torch.mul(x2 - x, y2 - y).float().unsqueeze(1)
        Q11 = torch.mul(weights, torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1).float().unsqueeze(1)
        Q12 = torch.mul(weights, torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1, y2 - y).float().unsqueeze(1)
        Q21 = torch.mul(weights, torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1).float().unsqueeze(1)
        Q22 = torch.mul(weights, torch.transpose(Q22, 0, 1))

        outputs = Q11 + Q21 + Q12 + Q22
        return outputs

class GraphProjectU(nn.Module):
    def __init__(self, use_z_weight=False):
        super(GraphProjectU, self).__init__()
        self.use_z_weight = use_z_weight
    
    def forward(self, vertices, img_feats, proj_mat):
        return self.project(vertices, img_feats, proj_mat)

    def project(self, point3d, img_feats, proj_mat):
        ones = torch.ones(point3d.shape[0], point3d.shape[1], 1).to(point3d.device)
        point4d = torch.cat([point3d, ones], -1)
        
        # normalized coord [-1., 1.]
        coords = torch.bmm(proj_mat, point4d.transpose(1, 2)).transpose(1,2)
        
        # trnasfer to image coord [0, 1]
        x = (coords[:,:, 0]/coords[:,:, 3] + 1) / 2
        y = 1 - (coords[:,:, 1]/coords[:,:, 3] + 1) / 2

        if self.use_z_weight:
            z = coords[:,:,2]/coords[:,:, 3]
            min_z, _ = z.min(1)
            min_z = min_z.unsqueeze(-1)
            max_z, _ = z.max(1)
            max_z = max_z.unsqueeze(-1)
            k = (min_z - z) / (max_z - min_z)
            z_weight = torch.exp(k)

        h, w = img_feats[0].shape[-2:]
        x0 = torch.clamp(x * w, 0, w-1)
        y0 = torch.clamp(y * h, 0, h-1)
        output = self.bilinear_interpolate(img_feats, x0, y0)

        if self.use_z_weight:
            output = z_weight.unsqueeze(-1) * output

        output = torch.cat([point3d, output], dim=-1)

        return output

    def bilinear_interpolate(self, img_feat, x, y):
        # image B * C * H * W
        B, C, H, W = img_feat.shape

        x1 = torch.floor(x).long()
        x2 = torch.ceil(x).long()
        y1 = torch.floor(y).long()
        y2 = torch.ceil(y).long()

        img_feat = img_feat.transpose(1, 3)
        Q11 = torch.stack([img_feat[b, x1[b], y1[b]] for b in range(B)], 0)
        Q12 = torch.stack([img_feat[b, x1[b], y2[b]] for b in range(B)], 0)
        Q21 = torch.stack([img_feat[b, x2[b], y1[b]] for b in range(B)], 0)
        Q22 = torch.stack([img_feat[b, x2[b], y2[b]] for b in range(B)], 0)

        x1 = x1.float()
        x2 = x2.float()
        y1 = y1.float()
        y2 = y2.float()

        weights = torch.mul(x2 - x, y2 - y).float().unsqueeze(-1).expand(-1, -1, C)
        Q11 = torch.mul(weights, Q11)

        weights = torch.mul(x2 - x, y - y1).float().unsqueeze(-1).expand(-1, -1, C)
        Q12 = torch.mul(weights, Q12)

        weights = torch.mul(x - x1, y2 - y).float().unsqueeze(-1).expand(-1, -1, C)
        Q21 = torch.mul(weights, Q21)

        weights = torch.mul(x - x1, y - y1).float().unsqueeze(-1).expand(-1, -1, C)
        Q22 = torch.mul(weights, Q22)

        outputs = Q11 + Q21 + Q12 + Q22
        return outputs


def compute_scale_project(img, proj_mat, verts):
    zeros = torch.zeros(img.shape[0], 1, 3).to(img.device)
    point3d = torch.cat([zeros, verts], dim=1)

    ones = torch.ones(point3d.shape[0], point3d.shape[1], 1).to(point3d.device)
    point4d = torch.cat([point3d, ones], -1)

    # normalized coord [-1., 1.]
    coords = torch.bmm(proj_mat, point4d.transpose(1, 2)).transpose(1,2)

    # trnasfer to image coord [0, 1]
    x = (coords[:,:, 0]/coords[:,:, 3] + 1) / 2
    y = 1 - (coords[:,:, 1]/coords[:,:, 3] + 1) / 2
    
    h, w = img[0].shape[-2:]
    x0 = x * w
    y0 = y * h
    img_coords = torch.stack([x0, y0], dim=-1)
    center = img_coords[:, 0]
    border = img_coords[:, 1:]

    scales = []

    B = img.shape[0]
    for b in range(B):
        mask = (torch.sum(img[b], dim=0) != 3.0).nonzero()
        dist0 = mask.float() - center[b]
        scale1 = torch.sqrt(torch.sum(dist0 * dist0, -1).max())

        dist1 = border[b] - center[b]
        scale2 = torch.sqrt(torch.sum(dist1 * dist1, -1).max())

        scales.append(scale1 / scale2)

    return torch.stack(scales, 0)