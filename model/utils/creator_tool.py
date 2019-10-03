import numpy as np
import numpy as cp
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from model.utils.nms import non_maximum_suppression

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.
        将(ground truth)地面实况(bounding boxes)边界框分配给给定的ROI。
        ProposalCreator产生2000个ROIS，再经过本ProposalTargetCreator的筛选产生128个用于自身的训练，规则如下:
            1. ROIS和GroundTruth_bbox的IOU大于0.5,选取一些(比如说本实验的32个)作为正样本
            2. 选取ROIS和GroundTruth_bbox的IOUS小于等于0（或者0.1）的选取一些比如说选取128-32=96个作为负样本
            然后分别对ROI_Headers进行训练

            为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）

    Args:
        n_sample (int): 要保留的iou数目
        pos_ratio (float): 作为正样本的比例
        pos_iou_thresh (float): 作为正样本的阈值
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.
            为抽样方案分配ground truth

        Args:
            roi (array):经过RPN网络的ProposalCreator的rois (R, 4)
            bbox (array): ground truth bounding boxes (R', 4)`.
            label (array): Ground truth bounding box labels (R',1 )`.
                            范围是[0, L - 1], L是类别数
            loc_normalize_mean (tuple of four floats): bouding boxes的均值
            loc_normalize_std (tupler of four floats): bounding boxes的方差

        Returns:

            * **sample_roi**: 再次经过下采样的n_sample个rois(S, 4) S= n_sample
            * **gt_roi_loc**: sample_roi与与之匹配最近的真实坐标框的偏差sample_roi(S, 4)`.
            * **gt_roi_label**: gt_roi_loc所匹配的真实坐标框的类别
                范围是[0, L - 1], L是类别数,O代表背景
        """
        n_bbox, _ = bbox.shape  # ground truth bounding boxes.

        roi = np.concatenate((roi, bbox), axis=0)  # 列变长,就是有更多的行,但是为什么要这样操作???????????

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)  # 若roi为[a,4],bbox为[b,4],则bbox_iou为[a,b]
        gt_assignment = iou.argmax(axis=1)  # 返回每行的最大值的索引
        max_iou = iou.max(axis=1)  # 返回每行的最大值
        gt_roi_label = label[gt_assignment] + 1
        """
        gt_assignment为[(roi.shape[0]+bbox.shape[0]),1],(roi为原始的,即形参)
        label为[bbox.shape[0],1],
        由于gt_assignment为每行的最大索引,故最大为num_truth_bounding_boxes.
        因此label[gt_assignment]是合理的,切返回值为[(roi.shape[0]+bbox.shape[0]),1]size的
        """

        # 筛选出大于前景IoU阈值的RoI,并随机抽取一定数量的pos_RoI
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)  # 随机选择大于阈值的pos_roi_per_this_image个anchor


        # 筛选出满足IoU阈值的负类RoI,并随机抽取一定数量的neg_RoI
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.
    将图像的事实 bounding boxes（边界框）分配给anchors 去训练faster_RCNN中的RPN网络

      Args:
        n_sample (int): 要产生坐标的数量
        pos_iou_thresh (float): 大于 IoU 的阈值即正类.
        neg_iou_thresh (float): 小于 IoU 的阈值即负类.
        pos_ratio (float): 正类的比例
    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.
        Args:
            bbox (array): 真实目标框的anchor box(R, 4)`.
            anchor (array): fearch map上所有的anchor (S, 4)`.
            img_size (tuple of ints): 图像的长和宽

        Returns:
            * **loc**: 为每个anchors与最大iou的truth bounding boxes的Offsets回归 (S, 4)`.
            * **label**: 所有anchor所匹配到最近真实框的label,其中1代表正类,0代表负类,-1表示不想关
        """

        img_H, img_W = img_size
        n_anchor = len(anchor)  # ==anchor.shape[0](anchor为原始的,因为后面有 _get_inside_index操作)
        inside_index = _get_inside_index(anchor, img_H, img_W)  # 有的anchor值比如H,为非法,即超过img_H, img_W,剔除它们,后面用_unmap进行恢复
        anchor = anchor[inside_index]  # anchor.shape[0] = len(inside_index)
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox) #不用传入inside_index
        # label: 根据iou标注有0, -1, 1的数组,此主要为了训练FPN
        # argmax_ious: 行最大索引, 每个anchor_box对应的最大iou的gt_anchor_box
        # label: 根据iou标注有0, -1, 1的数组

        loc = bbox2loc(anchor, bbox[argmax_ious])

        # 映射到原始坐标
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        """ 此函数创建label:
        计算gt_anchor_box和anchor_box的iou
        主要是对anchor_box操作的:
             默认所有anchor_box所对应的label为-1
             当anchor_box所匹配的最大iou的gt_anchor_box大于pos_iou_thresh时,anchor_box所对应的label为1
             当anchor_box所匹配的最大iou的gt_anchor_box小于neg_iou_thresh时,anchor_box所对应的label为0
             当gt_anchor_box所匹配的最大iou的anchor_box,anchor_box所对应的label为1
             当label为1的数量大于pos_ratio *n_sample时,需要在label为1的索引中随机选择pos_ratio *n_sample
             当label为0的数量大于neg_ratio *n_sample时,需要在label为0的索引中随机选择neg_ratio *n_sample

        :param inside_index:  不用传入inside_index参数:inside_index==anchor.shap[0]
        :param anchor: 在fearuth_map生成的W*H*9的anchor_box
        :param bbox:  为truth_anchor_box
        :return:
            argmax_ious: 行最大索引,每个anchor_box对应的最大iou的gt_anchor_box
            label: 根据iou标注有0,-1,1的数组
        """
        # label: 1 is positive, 0 is negative, -1 is dont care
        # 不用传入inside_index,inside_index==anchor.shape[0]
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1) #默认为-1

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)
        """
            argmax_ious: 行最大索引,每个anchor_box对应的最大iou的gt_anchor_box
            max_ious: 行最大值
            gt_argmax_ious: 列最大索引,每个gt_anchor_box对应的最大iou的anchor_box
        """

        # 低于neg_iou_thresh阈值的分为负类
        label[max_ious < self.neg_iou_thresh] = 0

        # 将所匹配的最大或者大于阈值的分为正类
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        # 如果多于一定数量,继续采样
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 如果多于一定数量,继续采样
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        """
        ious between the anchors and the gt boxes
        主要计算ious的行和列的最大值和索引:
        :param anchor: 在fearuth_map生成的W*H*9的anchor_box
        :param bbox:  为truth_anchor_box
        :param inside_index:  不用传入inside_index参数:inside_index==anchor.shap[0]
        :return:
            argmax_ious: 行最大索引
            max_ious: 行最大值
            gt_argmax_ious: 列最大索引
        """

        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)  # 行最大索引,即每个anchor对应的最大truth_anchor_box
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)  # 列最大索引,即每个truth_anchor_box对应的最大anchor
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        # 因为每个anchor_box只能对应一个最大iou的ruth_anchor_box,但是每个ruth_anchor_box可以对应多个最大iou的anchor_box
        # argmax函数只能返回第一个最大值的索引

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # 在 AnchorTargetCreator中一开始就进行了_get_inside_index操作,使得后面对于anchor的一系列操作都是对修改后的操作
    # 故现在根据_get_inside_index的inside_index将结果映射到原始anchor上

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        # 生成一个count行data.shape[1]列的empty矩阵,但是,but为什么要用emtpy,而不是最简单的np.ones((count,data.shape[1]))
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # 有的anchor值比如H为非法,即超过img_H, img_W,剔除它们
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    Args:
        nms_thresh (float): 调用NMS的boxes scored阈值
        n_train_pre_nms (int):  在训练阶段通过NMS前的top boxes scored个数
        n_train_post_nms (int): 在训练阶段通过NMS后的top boxes scored个数
        n_test_pre_nms (int): 在测试阶段通过NMS前的top boxes scored个数
        n_test_post_nms (int): 在测试阶段通过NMS后的top boxes scored个数
        force_cpu_nms (bool): 是否强制使用CPU模式,如果True则采用CPU mode,如果否则和输入类型一致
        min_size (int): 当小于这边参数时会默认舍弃该 anchors
    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """Propose RoIs.
        首先对roi = loc2bbox(anchor, loc)进行预处理,燃后对score进行排序,再选择pre_nms个top score进入NMS
        NMS阈值为nms_thresh,NMS后再选择前post_nms个roi作为ProposalCreator的返回值.
        Args:
            R : w*h*9
            loc (array): 预测anchors的偏移量比例。数据shape=(R, 4)
            score (array): 预测anchors的前景概率。数据shape=(R,).
            anchor (array): anchors坐标。数据shape=(R, 4).
            img_size (tuple of ints：H,W): 包含缩放后的图像大小.
            scale (float): 图像缩放比例.

        Returns:
            array:
                roi:proposal boxes坐标(array),数据shape=(S, 4):
                S在测试时间小于n_test_post_nms，在训练时间小于n_train_post_nms。
                S取决于预测边界框的大小和NMS丢弃的边界框的数量。

        """
        # NOTE: 在测试阶段,即faster_rcnn.eval(),需要设置self.traing = False,
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 通过base_anchor和loc解码获得目标anchor(即[y_min,x_min, y_max, x_max])

        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        #  slice(0, 4, 2 ) = [0,2]
        # np.clip(a,b,c) a为一根数组,b为min,c为max,夹逼
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # 删除预测的boxes长或者宽小于min_size*scale的boxes
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # score从高到低排序,选择前n_pre_nms个
        order = score.ravel().argsort()[::-1]  # 将score拉伸并逆序（从高到低）排序
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]  # 此时的roi的第一行就是score得分最高的那个anchor对应的anchor_boxes


        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        # 调用非极大值抑制函数，将重复的抑制掉，就可以将筛选后ROIS进行返回。
        # 经过NMS处理后Train数据集得到2000个框，Test数据集得到300个框
        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi
