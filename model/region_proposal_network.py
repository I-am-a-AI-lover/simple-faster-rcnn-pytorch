import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn


from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.
    RPN对vgg的conv5进行处理,然后提出关于truth的box

    Args:
        in_channels (int): 输入通道
        mid_channels (int): 中间矩阵的通道.
        ratios (list of floats):  anchors的长宽比 [0.5, 1, 2]
        anchor_scales (list of numbers):  anchors的尺寸[8, 16, 32]
        feat_stride (int): 进入RPN网络时的总步长. 16
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): ProposalCreator层的参数

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()

        # 生成一个(R, 4),R = len(anchor_scales)*len(ratios)=9,[y_min, x_min, y_max, x_max]
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)

        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # 2为背景和前景
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0) # 4为坐标
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.
        标注
        * N :批次
        * C :输入通道
        * H; W : 输入图像的尺寸
        * A: 每个像素的anchor数量:9
        Args:
            x (~torch.autograd.Variable):进入RPN网络的Features(N, C, H, W)`.
            img_size (tuple of ints): 经过scaling的height和width
            scale (float): The amount of scaling done to the input images after
                reading them from files.
        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):
            * **rpn_locs**: 对于anchors预测的bounding box offsets and scales (N, H W A, 4)
            * **rpn_scores**: 对于anchors预测的 foreground scores (N, H W A, 2)
            * **rois**:经过RPN中ProposalCreator后所propose的rois(将rpn_locs用loc2bbox函数转化后的anchor的一部分) (R,4).
            * **roi_indices**: 用來指示第几行的rois是第几个图片(批次中)propose的
            * **anchor**:生成的anchors(H W A, 4)`.
        """

        n, _, hh, ww = x.shape
        # _enumerate_shifted_anchor为由anchor_base得到整个feature_map的anchor, 即(9,4)->(h*w*9,4)
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            # proposal_layer为ProposalCreator类, 主要是为了根据rpn_fg_scores将rpn生成的roi区域减少
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index) #用来指示rois为第几个图片的roi

            rois = np.concatenate(rois, axis=0)  # rois为根据RPN 的loc与base_anchor解码的真实anchor坐标
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 为整个feature_map层生成anchors,anchor_base只适用于feature_map的(0,0)坐标
    # feat_stride  = 16
    # anchor_base 为默认anchor框

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride) # ==np.arange(height)*feat_stride
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    """
    meshgrid 为生成矩阵方格
    如若a = [1,2,3],b = [1,2,3]
    则 a,b = np.meshgrid(a,b)后
    a =  [[1,1,1],[2,2,2],[3,3,3]]
    b = [[1,2,3],[1,2,3],[1,2,3]]
    可以将a,b理解为x,y坐标
    """

    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0] # 为anchor的数量默认为9
    K = shift.shape[0] # 进入FPN网络的map的width*height
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    # 返回每个 feature map的cell(即width*height个cell)对应的映射anchor坐标
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    #xp = cuda.get_array_module(anchor_base)
    import torch as t
    import numpy as xp
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
