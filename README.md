## <center>Faster-R-CNN Pytorch中文注释 </center> ##

本文主要是对大牛的simple-faster-rcnn-pytorch的代码进行了一定的中文注释，仅仅为了更深理解two-stage的经典论文faster-rcnn，并没有对该代码进行测试，但在github有较多的star，故认为是比较优秀的代码



详见原始代码 [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/tree/dec293164d92702edc92225df21e371583637ef9 "Faster-R-CNN").

 ## Faster-R-CNN 整体流畅如下所示: ##

![uwOZcD.png](https://s2.ax1x.com/2019/10/03/uwOZcD.png)







Faster-R-CNN 主要由Vgg  extractor， RPN网络，RoI Head 三部分组成:

* Vgg  extractor 主要是基础提取特征，

* RPN 主要是propose anchor

* RoI Head 是针对RPN所propose的rois进行分类.




在train.py文件中实现了训练网络，但实际上的训练是在trainer.py上进行的.

在trainer.py中将Vgg  extractor， RPN， RoI Head整合在一起， 并在该函数中计算看loss



### RegionProposalNetwork类实现了RPN:



首先进行了４*W*H*A的loc矩阵预测，和 2*H*W*A的sorce矩阵预测

图中18指的是2＊9，2为前景和背景，9是9中anchor个数，36指4＊9，4是坐标偏差.

RPN利用 `AnchorTargetCreator`自身训练的同时，还会提供RoIs(region of interests)给RoIHead作为训练样本。

然后进行ProposalCreator操作，主要为了筛选较为合格的RoIs

* 对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。
* 选取概率较大的12000个anchor
* 利用回归的位置参数，修正这12000个anchor的位置，得到RoIs
* 利用非极大值NMS抑制，选出概率最大的2000个RoIs



RoIs为根据RPN 的loc与base_anchor用`loc2bbox`解码的真实anchor坐标．对sorce进行从大到小的排序，然后选择一满足一定数量和nms阈值条件sorce对应的的RoIs往下传递．



<u>**RoIs:贯穿全文，不断筛选减少**</u>



整个网络返回：

```
rpn_locs: 对于anchors预测的bounding box offsets and scales (N，H＊W＊A，4)
rpn_scores: 对于anchors预测的 foreground scores (N，H＊W＊A，2)
RoIs: 经过RPN中ProposalCreator后所propose的RoIs(即进入ProposalCreator后的anchor的一部分) (R，4)
roi_indices: 用來指示第几行的RoIs是第几个图片(批次中)propose的
anchor:生成的anchors(H＊W＊A，4).
```



### RPN进入RoI Head中间进行了ProposalTargetCreator

RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用ProposalTargetCreator 选择128个RoIs用以训练．该操作主要将RPN后的RoIs再次进行减少和根据IOU进行label标注，以获得RoI Head的cls_loss的回归值

规则如下:   

1. RoIs和GroundTruth_bbox的IOU大于0.5，选取一些(比如说本实验的32个)作为正样本 
2. 选取RoIs和GroundTruth_bbox的IOUS小于等于0（或者0.1）的选取一些比如说选取128-32=96个作为负样本，然后分别对ROI_Headers进行训练

返回如下:

```
sample_roi: 再次经过下采样的n_sample个RoIs(S， 4) S= n_sample
gt_roi_loc: sample_roi与与之匹配最近的真实坐标框的偏差sample_roi(S， 4)
gt_roi_label: gt_roi_loc所匹配的真实坐标框的类别
```



`VGG16RoIHead`类实现了RoI Head
在该类中输入为ProposalTargetCreator的输出，首先进行了ROI Pooling操作，将进入RPN网络的feature_map映射到RoIs区域，即从feature_map"扣"出RoIs区域，然后进行分类和回归．



loss由两种组成，分别是loc_loss和cls_loss，且RPN和，RoI Head都会生成loc_loss和cls_loss，故共四种loss


 `AnchorTargetCreator`类

计算base_anchor与真实实例box的iou，按照一定阈值条件选择一定数量的正类和负类和所有base_anchor与最近真实实例box的回归.

将20000多个候选的anchor选出256个anchor进行二分类和所有的anchor进行回归位置 。选择方式如下：

* 对于每一个ground truth bounding box (gt_bbox)，选择和它IoU最高的一个anchor作为正样本。
* 对于剩下的anchor，从中选择和任意一个gt_bbox重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
* 随机选择和gt_bbox重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256．



返回：

argmax_ious: 行最大索引,每个anchor_box对应的最大iou的gt_anchor_box
label: 根据iou标注有{0,-1,1}的数组



在计算RPN_loss时

只计算正类的损失，由于RPN只会生成真实实例box的坐标，不会区分类别，故在RPN的cls_loss计算中，只会计算两种分类结果的分类损失(二分类)，即正类和负类，不区分是哪个具体正类类别．在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。RPN_loss 的计算主要为了精确RPN的精度.



在计算ROI_loss时.

 主要是主要是网络最终分类和回归的损失，其中真实坐标是在ProposalTargetCreator时产生的带有类别分类的标签



大部分核心代码有注释，除了`nms`和`ROI Pooling`等.

对于原代码,本人删除了部分英文注释,**并将部分导入语句**

原始README.MD文件改名为**README1.MD**

```python
import cupy as xp
# 改为 
import numpy as xp
```



本人阅读顺序大致为

```
{train.py, trainer.py}, /model/utils/{bbox_tools.py, creator_tools.py}, /model/{faster_rcnn.py, faster_rcnn_vgg16.py, region_proposal_network.py, roi_module.py}
```



