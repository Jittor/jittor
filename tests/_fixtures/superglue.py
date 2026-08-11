from copy import deepcopy
from pathlib import Path
import jittor as jt
import jittor.nn as nn
import numpy as np
import os

split_size = 1000000

conv_opt = int(os.environ.get("conv_opt", "0"))

if conv_opt:
    Conv1d_sp = nn.Conv1d_sp
else:
    Conv1d_sp = nn.Conv1d


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(Conv1d_sp(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))
                # layers.append(nn.LayerNorm(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    size = image_shape.flip(1)  # shape=(b,2) ;h w -> w, h
    center = size / 2
    scaling = size.float32().max(1, keepdims=True) * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers, keypoint_position_dim=2):
        super().__init__()
        # self.keypoint_position_dim = keypoint_position_dim
        self.encoder = MLP([keypoint_position_dim + 1] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def execute(self, kpts, scores):
        inputs = jt.concat([kpts.t(), scores.unsqueeze(1)], dim=1)
        return self.encoder(inputs)

cnt = 0

def attention(query, key, value):
    global cnt
    cnt += 1
    b, d, h, n = query.shape
    # print("attention", b,d,h,n, cnt)
    dim_factor = (1.0 / d)**0.5
    query = query.transpose(0, 2, 3, 1).reshape(b * h, -1, d) * dim_factor
    key = key.transpose(0, 2, 1, 3).reshape(b * h, d, -1)
    value = value.transpose(0, 2, 3, 1).reshape(b * h, -1, d)
    # print("attention", query.shape, key.shape, value.shape)

    data = []
    for i in range(0, query.shape[0], split_size):
        end = min(i + split_size, query.shape[0])
        tmp1 = nn.bmm(query[i:end], key[i:end])
        tmp2 = nn.softmax(tmp1, dim=-1)
        tmp3 = nn.bmm(tmp2, value[i:end])
        tmp3.sync()
        data.append(tmp3)
    tmp3 = jt.concat(data)
    
    # for i in range(0, query.shape[0], split_size):
    #     end = min(i + split_size, query.shape[0])
    #     tmp1 = nn.bmm(query[:,i:end], key[:,i:end])
    #     tmp2 = nn.softmax(tmp1, dim=-1)
    #     tmp3 = nn.bmm(tmp2, value[:,i:end])
    #     tmp3.sync()
    #     data.append(tmp3)
    # tmp3 = jt.concat(data, dim=1)

    # tmp1 = nn.bmm(query, key)
    # print(tmp1.shape)
    # tmp2 = nn.softmax(tmp1, dim=-1)
    # print(tmp2.shape)
    # tmp3 = nn.bmm(tmp2, value)
    # print(tmp3.shape)
    return tmp3.reshape(b, h, -1, d).transpose(0, 3, 1, 2)
    return nn.bmm(nn.softmax(nn.bmm(query, key), dim=-1), value).reshape(b, h, -1, d).transpose(0, 3, 1, 2)


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = Conv1d_sp(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def execute(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).reshape(batch_dim, self.dim, self.num_heads, -1) for l, x in zip(self.proj, (query, key, value))]
        x = attention(query, key, value)
        # x = attention_chunk(query, key, value)
        return self.merge(x.reshape(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def execute(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(jt.concat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))])
        self.is_cross = [x == 'cross' for x in layer_names]

    def execute(self, desc0, desc1):
        for layer, is_cross in zip(self.layers, self.is_cross):
            layer.attn.prob = []
            if is_cross:
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            # delta0, delta1 = layer(desc0, src0), layer(desc1, src1)

            delta0 = layer(desc0, src0)
            # print(delta0.numel()*4)
            # breakpoint()
            jt.sync_all()
            delta1 = layer(desc1, src1)
            jt.sync_all()
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            jt.sync_all()
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = jt.zeros_like(log_mu), jt.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - (Z + v.unsqueeze(1)).exp().sum(dim=2).log()
        v = log_nu - (Z + u.unsqueeze(2)).exp().sum(dim=1).log()
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    ms, ns = jt.float(m, requires_grad=False), jt.float(n, requires_grad=False)

    bins0 = alpha.broadcast([b, m, 1])
    bins1 = alpha.broadcast([b, 1, n])
    alpha = alpha.broadcast([b, 1, 1])

    couplings = jt.concat([jt.concat([scores, bins0], -1), jt.concat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = jt.concat([norm.broadcast([m]), ns.log() + norm])
    log_nu = jt.concat([norm.broadcast([n]), ms.log() + norm])
    log_mu, log_nu = log_mu[None].broadcast([b, m + 1]), log_nu[None].broadcast([b, n + 1])

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return jt.ones(x.shape[dim], dtype=x.dtype)[None].cumsum()[0] - 1  # traceable in 1.1


default_config = {
    'descriptor_dim': 256,  # SuperPoint
    'weights': 'indoor',
    'keypoint_encoder': [32, 64, 128, 256],  # SuperPoint
    'GNN_layers': ['self', 'cross'] * 9,
    'sinkhorn_iterations': 100,
    'match_threshold': 0.2,
}


def get_weighted_loss_batch(scores, all_matches):
    matches0, matches1 = all_matches.chunk(chunks=2, dim=2)
    batchIdx = jt.arange(all_matches.shape[0]).unsqueeze(1).repeat(1, all_matches.shape[1])
    batchIdx, matches0, matches1 = batchIdx.view(-1), matches0.view(-1), matches1.view(-1)
    valid_index0, valid_index1 = matches0 >= 0, matches1 >= 0
    valid_match = jt.logical_and(valid_index0, valid_index1)
    valid_unmatch = jt.logical_xor(valid_index0, valid_index1)
    num_match = valid_match.sum().maximum(1e-9)
    num_unmatch = valid_unmatch.sum().maximum(1e-9)



    score_ = scores[batchIdx, matches0, matches1]
    score_match_ = (score_*valid_match).float32().sum() / num_match
    score_umatch_ = (score_*valid_unmatch).float32().sum() / num_unmatch
    return -(num_unmatch * score_match_ + num_match * score_umatch_) / (num_match + num_unmatch)
    # print(score_umatch_, score_match_)
    # return -(score_match + score_umatch) / (num_match + num_unmatch)

    score_match = scores[(batchIdx[valid_match], matches0[valid_match], matches1[valid_match])].float32().mean() if num_match > 0 else 0
    score_umatch = scores[(batchIdx[valid_unmatch], matches0[valid_unmatch], matches1[valid_unmatch])].float32().mean() if num_unmatch > 0 else 0
    # print(score_match, score_umatch)
    return -(num_unmatch * score_match + num_match * score_umatch) / (num_match + num_unmatch)


def add_dustbin(scores, alpha):
    b, m, n = scores.shape
    bins0 = jt.broadcast(alpha, (b, m, 1))
    bins1 = jt.broadcast(alpha, (b, 1, n))
    alpha = jt.broadcast(alpha, (b, 1, 1))
    couplings = jt.concat([jt.concat([scores, bins0], -1), jt.concat([bins1, alpha], -1)], 1)
    return couplings


class SuperGlue(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = {**default_config, **config}
        self.descriptor_dim = config['descriptor_dim']
        self.keypoint_encoder = config['keypoint_encoder']
        self.GNN_layers = config['GNN_layers']
        self.sinkhorn_iterations = config['sinkhorn_iterations']
        self.match_threshold = config['match_threshold']
        self.keypoint_position_dim = config['keypoint_position_dim']
        self.use_dual_softmax = config['use_dual_softmax']
        self.scale = jt.float(self.descriptor_dim**-0.5).stop_grad()
        # self.scale.requires_grad = False

        # self.des_extend = MLP([128, 256])

        self.kenc = KeypointEncoder(self.descriptor_dim, self.keypoint_encoder, keypoint_position_dim=self.keypoint_position_dim)

        self.gnn = AttentionalGNN(self.descriptor_dim, self.GNN_layers)

        self.final_proj = Conv1d_sp(self.descriptor_dim, self.descriptor_dim, kernel_size=1, bias=True)

        self.bin_score = jt.float(1.0)

    def execute(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        all_matches = data['all_matches']
        # match_num = data['match_num']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0 or all_matches.shape[1] == 0:  # no keypoints or no matches/unmatches
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': jt.ones(shape0, dtype=jt.int),
                'matches1': jt.ones(shape1, dtype=jt.int),
                'matching_scores0': jt.zeros(shape0, dtype=jt.float),
                'matching_scores1': jt.zeros(shape1, dtype=jt.float),
                'skip_train': True
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['shape0'])
        kpts1 = normalize_keypoints(kpts1, data['shape1'])

        # Keypoint MLP encoder.
        # desc0 = self.des_extend(desc0) + self.kenc(kpts0, data['scores0'])
        # desc1 = self.des_extend(desc1) + self.kenc(kpts1, data['scores1'])
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        desc0, desc1 = self.final_proj(desc0), self.final_proj(desc1)
        desc0_t = desc0.t()
        losses = []

        for i in range(0, desc1.shape[0], split_size):
            end = min(desc1.shape[0], i + split_size)

            # Compute matching descriptor distance.
            scores = nn.bmm(desc0_t[i:end], desc1[i:end]) * self.scale  # 457.76 MB
            scores.sync()

            # Run the optimal transport.
            if self.use_dual_softmax:
                scores = add_dustbin(scores, self.bin_score)  # 458.68 MB
                scores.sync()
                dual_softmax0, dual_softmax1 = nn.log_softmax(scores, 1), nn.log_softmax(scores, 2)
                scores = dual_softmax0 + dual_softmax1  # 458.22 MB
                scores.sync()
            else:
                scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

            # loss = torch.stack([get_match_score(scores[b], all_matches[b]) for b in range(all_matches.shape[0])])

            loss = get_weighted_loss_batch(scores, all_matches[i:end])
            loss.sync()
            losses.append(loss)
        loss = jt.concat(losses)
        '''
        # Compute matching descriptor distance.
        scores = nn.bmm(desc0.t(), desc1) * self.scale # 457.76 MB
        scores.sync()

        # Run the optimal transport.
        if self.use_dual_softmax:
            scores = add_dustbin(scores, self.bin_score) # 458.68 MB
            scores.sync()
            dual_softmax0, dual_softmax1 = nn.log_softmax(scores, 1), nn.log_softmax(scores, 2)
            scores = dual_softmax0 + dual_softmax1 # 458.22 MB
            scores.sync()
        else:
            scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        # loss = torch.stack([get_match_score(scores[b], all_matches[b]) for b in range(all_matches.shape[0])])

        loss = get_weighted_loss_batch(scores, all_matches)
        # print(scores.shape, all_matches.shape, loss.shape)
        '''

        # matches0, matches1 = all_matches.chunk(chunks=2, dim=2)
        # batchIdx = jt.arange(0, b).unsqueeze(1).repeat(1, num)
        # batchIdx, matches0, matches1 = batchIdx.view(-1), matches0.view(-1), matches1.view(-1)
        # validmatch = (matches0 >= 0) | (matches1 >= 0)
        # batchIdx, matches0, matches1 = batchIdx[validmatch], matches0[validmatch], matches1[validmatch]
        # matches0[matches0 == -1] = n
        # matches1[matches1 == -1] = m
        # loss_mean = -scores[(batchIdx, matches0, matches1)].mean()
        # loss_mean = nn.l1_loss(loss_mean, jt.float(0.0))

        if not data['return_match']:
            return {'loss': loss}

        with jt.no_grad():
            b, n, m = scores.shape
            # Get the matches with score above "match_threshold".
            indices0, max0 = scores[:, :-1, :-1].argmax(2)
            indices1, max1 = scores[:, :-1, :-1].argmax(1)
            mutual0 = jt.arange(0, n)[None] == indices1.gather(1, indices0)
            mutual1 = jt.arange(0, m)[None] == indices0.gather(1, indices1)
            # zero = scores.new_tensor(0)
            # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores0 = max0.exp()
            mscores0[mutual0.logical_not()] = 0
            # mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            mscores1 = mscores0.gather(1, indices1)
            mscores1[mutual1.logical_not()] = 0
            valid0 = mutual0 & (mscores0 > self.match_threshold)
            valid1 = mutual1 & valid0.gather(1, indices1)
            # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
            indices0[valid0.logical_not()] = -1
            indices1[valid1.logical_not()] = -1

        return {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'loss': loss,
        }

        # scores big value or small value means confidence? log can't take neg value