import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import numbers
from torch._six import container_abcs
from itertools import repeat as repeat_


class GraphProjectUPoint(nn.Module):
    def __init__(self, use_z_weight=False):
        super(GraphProjectUPoint, self).__init__()
        self.use_z_weight = use_z_weight

    def forward(self, vertices, img_feats, proj_mat):
        return self.project(vertices, img_feats, proj_mat)

    def project(self, point3d, img_feats, proj_mat):
        ones = torch.ones(point3d.shape[0], point3d.shape[1], 1).to(point3d.device)
        point4d = torch.cat([point3d, ones], -1)

        # normalized coord [-1., 1.]
        coords = torch.bmm(proj_mat, point4d.transpose(1, 2)).transpose(1, 2)

        # trnasfer to image coord [0, 1]
        x = (coords[:, :, 0] / coords[:, :, 3] + 1) / 2
        y = 1 - (coords[:, :, 1] / coords[:, :, 3] + 1) / 2

        if self.use_z_weight:
            z = coords[:, :, 2] / coords[:, :, 3]
            min_z, _ = z.min(1)
            min_z = min_z.unsqueeze(-1)
            max_z, _ = z.max(1)
            max_z = max_z.unsqueeze(-1)
            k = (min_z - z) / (max_z - min_z)
            z_weight = torch.exp(k)

        h, w = img_feats[0].shape[-2:]
        x0 = torch.clamp(x * w, 0, w - 1)
        y0 = torch.clamp(y * h, 0, h - 1)
        output = self.bilinear_interpolate(img_feats, x0, y0)

        if self.use_z_weight:
            output = z_weight.unsqueeze(-1) * output

        # output = torch.cat([point3d, output], dim=-1)
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


class BlockFC(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BlockFC, self).__init__()
        self.fc_1 = nn.Linear(in_planes, out_planes)
        self.fc_2 = nn.Linear(out_planes, out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        return x


class PointCloudEncoder(nn.Module):
    def __init__(self, param):
    # def __init__(self):
        super(PointCloudEncoder, self).__init__()
        down_planes = [64, 128, 256, 512]
        self.param = param

        # projection
        self.project_point = GraphProjectUPoint(param.use_z_weight)
        # self.project_point = GraphProjectUPoint(True)

        # adain
        self.block_0 = BlockFC(3, down_planes[0])   # 3->64
        self.block_fc_0 = nn.Linear(down_planes[0], down_planes[0])  # adain
        self.block_1 = BlockFC(down_planes[0], down_planes[1])   # 64->128
        self.block_fc_1 = nn.Linear(down_planes[1], down_planes[1])  # adain
        self.block_2 = BlockFC(down_planes[1], down_planes[2])   # 128->256
        self.block_fc_2 = nn.Linear(down_planes[2], down_planes[2])  # adain
        self.block_3 = BlockFC(down_planes[2], down_planes[3])   # 256->512
        self.block_fc_3 = nn.Linear(down_planes[3], down_planes[3])  # adain
        self.relu = nn.ReLU()

    def dimshuffle(self, x, pattern):
        no_expand_pattern = [x for x in pattern if x != 'x']
        y = x.permute(*no_expand_pattern)
        shape = list(y.shape)
        for idx, e in enumerate(pattern):
            if e == 'x':
                shape.insert(idx, 1)
        return y.view(*shape)

    def transform(self, pc_feat, img_feat, fc):
        pc_feat = (pc_feat - torch.mean(pc_feat, -1, keepdim=True)) / torch.sqrt(torch.var(pc_feat, -1, keepdim=True) + 1e-8)
        mean, var = torch.mean(img_feat, (2, 3)), torch.var(torch.flatten(img_feat, 2), 2)
        output = (pc_feat + self.dimshuffle(mean, (0, 'x', 1))) * torch.sqrt(self.dimshuffle(var, (0, 'x', 1)) + 1e-8)
        return fc(output)

    def forward(self, points, img_feats, proj_mat):
        features = []
        feat, feat2_3, feat3_3, feat4_3, feat5_4 = img_feats  # 1008 64 128 256 512
        # print(feat.shape, feat2_3.shape, feat3_3.shape, feat4_3.shape, feat5_4.shape)
        if self.param.pcdnet_adain:
        # if True:
            pc_feat = self.block_0(points)
            features += [self.transform(pc_feat, feat2_3, self.block_fc_0)]
            pc_feat = self.block_1(pc_feat)
            features += [self.transform(pc_feat, feat3_3, self.block_fc_1)]
            pc_feat = self.block_2(pc_feat)
            features += [self.transform(pc_feat, feat4_3, self.block_fc_2)]
            pc_feat = self.block_3(pc_feat)
            features += [self.transform(pc_feat, feat5_4, self.block_fc_3)]

        # projection
        features += [self.project_point(points, feat2_3, proj_mat)]
        features += [self.project_point(points, feat3_3, proj_mat)]
        features += [self.project_point(points, feat4_3, proj_mat)]
        features += [self.project_point(points, feat5_4, proj_mat)]

        # originate points
        features += [points]

        features = torch.cat(features, dim=-1)
        return features


class PointCloudDecoder(nn.Module):
    def __init__(self, param):
    # def __init__(self):
        super(PointCloudDecoder, self).__init__()
        self.param = param
        self.relu = nn.ReLU()

        if self.param.pcdnet_adain:
        # if True:
            self.fc1 = nn.Linear(1923, 512)
        else:
            self.fc1 = nn.Linear(963, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x, points):
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))
        x = self.fc4(self.relu(x))    # 2048 * 3
        return x + points


def dimshuffle(x, pattern):
    assert isinstance(pattern, (list, tuple)), 'pattern must be a list/tuple'
    no_expand_pattern = [x for x in pattern if x != 'x']
    y = x.permute(*no_expand_pattern)
    shape = list(y.shape)
    for idx, e in enumerate(pattern):
        if e == 'x':
            shape.insert(idx, 1)
    return y.view(*shape)


class DimShuffle(nn.Module):
    def __init__(self, pattern, input_shape=None):
        super(DimShuffle, self).__init__()
        self.pattern = pattern

    def forward(self, input):
        return dimshuffle(input, self.pattern)


def _make_input_shape(m, n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x

        return tuple(repeat_(None, m)) + (x, ) + tuple(repeat_(None, n))
    return parse


_pointset_shape = _make_input_shape(2, 0)


class GraphXConv(nn.Module):
    def __init__(self, input_shape, out_features, out_instances=None, rank=None, bias=True,
                 weights_init=None, bias_init=None, **kwargs):
        input_shape = _pointset_shape(input_shape)
        super(GraphXConv, self).__init__()
        self.input_shape = input_shape
        self.out_features = out_features
        self.out_instances = out_instances if out_instances else input_shape[-2]
        if rank:
            assert rank <= self.out_instances // 2, 'rank should be smaller than half of num_out_points'

        self.rank = rank
        self.activation = nn.ReLU()
        pattern = list(range(len(input_shape)))
        pattern[-1], pattern[-2] = pattern[-2], pattern[-1]
        self.pattern = pattern
        self.weights_init = weights_init
        self.bias_init = bias_init

        self.weight = nn.Parameter(torch.Tensor(out_features, input_shape[-1]))
        if self.rank is None:
            self.mixing = nn.Parameter(torch.Tensor(self.out_instances, input_shape[-2]))
        else:
            self.mixing_u = nn.Parameter(torch.Tensor(self.rank, input_shape[-2]))
            self.mixing_v = nn.Parameter(torch.Tensor(self.out_instances, self.rank))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.mixing_bias = nn.Parameter(torch.Tensor(self.out_instances))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mixing_bias', None)

        self.reset_parameters()

    def forward(self, input):
        output = dimshuffle(input, self.pattern)
        mixing = torch.mm(self.mixing_v, self.mixing_u) if self.rank else self.mixing
        output = torch.matmul(output, mixing.t())
        if self.mixing_bias is not None:
            output = output + self.mixing_bias

        output = dimshuffle(output, self.pattern)
        output = torch.matmul(output, self.weight.t())
        if self.bias is not None:
            output = output + self.bias

        return self.activation(output)

    def output_shape(self):
        return self.input_shape[:-2] + (self.out_instances, self.out_features)

    def reset_parameters(self):
        weights_init = partial(nn.init.kaiming_uniform_, a=np.sqrt(5)) if self.weights_init is None \
            else self.weights_init

        weights_init(self.weight)
        if self.rank:
            weights_init(self.mixing_u)
            weights_init(self.mixing_v)
        else:
            weights_init(self.mixing)

        if self.bias is not None:
            if self.bias_init is None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                bias_init = partial(nn.init.uniform_, a=-bound, b=bound)
            else:
                bias_init = self.bias_init

            bias_init(self.bias)
            nn.init.zeros_(self.mixing_bias)

    def extra_repr(self):
        s = '{input_shape}, out_features={out_features}, out_instances={out_instances}, rank={rank}'
        s = s.format(**self.__dict__)
        s += ', activation={}'.format(self.activation.__name__)
        return s


# GraphX
class PointCloudGraphXDecoder(nn.Module):
    def __init__(self, input_shape):
        super(PointCloudGraphXDecoder, self).__init__()
        self.input_shape = input_shape
        B, T, D = input_shape
        self.graphXConv1 = GraphXConv(self.input_shape, 512)
        self.graphXConv2 = GraphXConv((B, T, 512), 256)
        self.graphXConv3 = GraphXConv((B, T, 256), 128)
        self.fc4 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x, points):
        x = self.graphXConv1(x)
        x = self.graphXConv2(self.relu(x))
        x = self.graphXConv3(self.relu(x))
        x = self.fc4(self.relu(x))
        return x + points


class ResGraphXConv(nn.Module):
    def __init__(self, input_shape, out_features, num_instances=None):
        super(ResGraphXConv, self).__init__()
        self.input_shape = input_shape
        B, T, D = input_shape
        input_shape_2 = (B, T, out_features)
        self.out_features = out_features
        self.relu = nn.ReLU()

        self.fc_graphXConv = nn.Sequential(
            nn.Linear(input_shape[-1], out_features),
            nn.ReLU(),
            GraphXConv(input_shape_2, out_features, out_instances=num_instances)
        )
        if num_instances is None:
            self.res = (lambda x: x) if (out_features == input_shape[-1]) \
                else nn.Linear(input_shape[-1], out_features)
        else:
            self.res = GraphXConv(input_shape, out_features, out_instances=num_instances)

    def forward(self, input):
        res = self.res(input)
        output = self.fc_graphXConv(input) + res
        return self.relu(output)


# resGraphX
class PointCloudResGraphXDecoder(nn.Module):
    def __init__(self, input_shape):
        super(PointCloudResGraphXDecoder, self).__init__()
        self.input_shape = input_shape
        B, T, D = input_shape
        self.resGraphXConv1 = ResGraphXConv(self.input_shape, 512)
        self.resGraphXConv2 = ResGraphXConv((B, T, 512), 256)
        self.resGraphXConv3 = ResGraphXConv((B, T, 256), 128)
        self.fc4 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x, points):
        x = self.resGraphXConv1(x)
        x = self.resGraphXConv2(self.relu(x))
        x = self.resGraphXConv3(self.relu(x))
        x = self.fc4(self.relu(x))
        return x + points


# # resGraphXUp  may have bug
# class PointCloudResGraphXUpDecoder(nn.Module):
#     def __init__(self, input_shape):
#         super(PointCloudResGraphXUpDecoder, self).__init__()
#         self.input_shape = input_shape
#         B, T, D = input_shape
#         self.resGraphXUpConv1 = ResGraphXConv(self.input_shape, 512, num_instances=self.input_shape[1] * 2)
#         output_shape = (B, T, 512)
#         self.resGraphXUpConv2 = ResGraphXConv(output_shape, 256, num_instances=output_shape[1] * 2)
#         output_shape = (B, T, 256)
#         self.resGraphXUpConv3 = ResGraphXConv(output_shape, 128, num_instances=output_shape[1] * 2)
#         self.fc4 = nn.Linear(128, 3)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.resGraphXUpConv1(self.relu(x))
#         x = self.resGraphXUpConv2(self.relu(x))
#         x = self.resGraphXUpConv3(self.relu(x))
#         x = self.fc4(self.relu(x))
#         return x


class ResFC(nn.Module):
    def __init__(self, input_shape, out_features, num_instances=None):
        super(ResFC, self).__init__()
        self.input_shape = input_shape
        B, T, D = input_shape
        input_shape_2 = (B, T, out_features)
        self.out_features = out_features
        self.relu = nn.ReLU()
        self.fc_graphXConv = nn.Sequential(
            nn.Linear(input_shape[-1], out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        if num_instances is None:
            self.res = (lambda x: x) if (out_features == input_shape[-1]) \
                else nn.Linear(input_shape[-1], out_features)
        else:
            self.res = GraphXConv(input_shape, out_features, out_instances=num_instances)

    def forward(self, input):
        res = self.res(input)
        output = self.fc_graphXConv(input) + res
        return self.relu(output)


# graphX Res
class PointCloudResDecoder(nn.Module):
    def __init__(self, input_shape):
        super(PointCloudResDecoder, self).__init__()
        self.input_shape = input_shape
        B, T, D = input_shape
        self.resFc1 = ResFC(self.input_shape, 512)
        self.resFc2 = ResFC((B, T, 512), 256)
        self.resFc3 = ResFC((B, T, 256), 128)
        self.fc4 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x, points):
        x = self.resFc1(x)
        x = self.resFc2(self.relu(x))
        x = self.resFc3(self.relu(x))
        x = self.fc4(self.relu(x))
        return x + points


# def create_encoder(encoder):
#     pass


def create_decoder(param, pc_feats_shape):
# def create_decoder(decoder, pc_feats_shape):
    decoder = param.decoder
    # pc_decoder = None
    if decoder == 'PC_Dec':
        # print("use: PointCloudDecoder()")
        pc_decoder = PointCloudDecoder(param)
    elif decoder == 'PC_ResDec':
        # print("use: PointCloudResDecoder()")
        pc_decoder = PointCloudResDecoder(pc_feats_shape)
    elif decoder == 'PC_GraphXDec':
        # print("use: PointCloudGraphXDecoder()")
        pc_decoder = PointCloudGraphXDecoder(pc_feats_shape)
    elif decoder == 'PC_ResGraphXDec':
        # print("use: PointCloudResGraphXDecoder()")
        pc_decoder = PointCloudResGraphXDecoder(pc_feats_shape)
    # elif decoder == 'PC_ResGraphXUpDec':
    #     print("use: PointCloudResGraphXUpDecoder()")
    #     pc_decoder = PointCloudResGraphXUpDecoder(pc_feats_shape)
    return pc_decoder


class PointCloudDeformNet(nn.Module):
    def __init__(self, param):
    # def __init__(self):
        super(PointCloudDeformNet, self).__init__()
        self.param = param
        B = param.batch_size
        T = param.sample_pcl_num
        if param.pcdnet_adain:
            D = 1923
        else:
            D = 963
        pc_feats_shape = (B, T, D)
        self.pointcloud_encoder = PointCloudEncoder(param)
        self.pointcloud_decoder = create_decoder(param, pc_feats_shape)

        # B = 2
        # D = 1923
        # T = 2048
        # pc_feats_shape = (B, T, D)
        # self.pointcloud_encoder = PointCloudEncoder()
        # self.pointcloud_decoder = create_decoder(decoder='PC_ResGraphXDec', pc_feats_shape=pc_feats_shape)

    def forward(self, img_feats, points, proj_mat):
        pc_feats = self.pointcloud_encoder(points, img_feats, proj_mat)
        output_points = self.pointcloud_decoder(pc_feats, points)
        return output_points


if __name__ == '__main__':
    from image_model import VGGNet

    img = torch.randn(2, 3, 224, 224)
    vgg = VGGNet()
    img_feats = vgg(img)

    points = torch.rand(2, 2048, 3)
    proj_mat = torch.rand(2, 4, 4)

    # pointcloud_encoder = PointCloudEncoder()
    # pc_feats = pointcloud_encoder(points, img_feats, proj_mat)
    # print(pc_feats.shape)
    # pc_feats_shape = pc_feats.shape
    #
    # # pointcloud_decoder = PointCloudDecoder()
    # # pointcloud_decoder = PointCloudGraphXDecoder(pc_feats_shape)
    # # pointcloud_decoder = PointCloudResGraphXDecoder(pc_feats_shape)
    # pointcloud_decoder = PointCloudResDecoder(pc_feats_shape)
    #
    # # pointcloud_decoder = PointCloudResGraphXUpDecoder(pc_feats_shape)  # may have bug
    # output_points = pointcloud_decoder(pc_feats)

    pcdnet = PointCloudDeformNet()
    output_points = pcdnet(img_feats, points, proj_mat)

    print(output_points.shape)
