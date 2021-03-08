# Collection of Image Recognition and Object Detection related papers


## [Content](#content)
<table>
<tr><td colspan="3"><a href="#surveys">0. Surveys</a></td></tr>
<tr><td colspan="3"><a href="#image-recognition">1. Image Recognition</a></td></tr>
<tr>
    <td>&ensp;<a href="#transformers">1.3 Transformers</a></td>
    <td>&ensp;<a href="#gnn">1.3 GNN</a></td>
</tr>
<tr><td colspan="3"><a href="#object-detection">2. Object Detection </a></td></tr>
<tr>
    <td>&emsp;<a href="#anchor-based-one-stage">2.11 Anchor-Based: one-stage</a></td>
    <td>&emsp;<a href="#anchor-based-two-stage">2.12 Anchor-Based: two-stage</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#anchor-free-keypoint-based">2.21 Anchor-Free: Keypoint-based</a></td>
    <td>&ensp;<a href="#anchor-free-center-based">2.22 Anchor-Free: Center-based</a></td>
</tr>
<tr>
    <td>&ensp;<a href="#transformers">2.3 Transformers</a></td>
    <td>&ensp;<a href="#gnn">2.4 GNN</a></td>
</tr>
<tr><td colspan="3"><a href="#architecture">2. Architecture </a></td></tr>
<tr><td colspan="3"><a href="#miscellaneous">2. Miscellaneous </a></td></tr>

</table>

## [Surveys](#content)
* **Deep Learning for Generic Object Detection: A Survey.** IJCV, 2019. <br/>
  [[Paper](https://link.springer.com/article/10.1007/s11263-019-01247-4)]<br/>  
* **Transformers in Vision: A Survey.** arXiv, 2021. <br/>
  [[Paper](https://arxiv.org/pdf/2101.01169.pdf)]<br/>


## [Image Recognition](#content)
* **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** ICLR, 2021. <br/>
  [[Paper](https://arxiv.org/pdf/2010.11929.pdf), [Code & pre-trained models](https://github.com/google-research/vision_transformer)]<br/>


## [Object Detection](#content)
### [Anchor-Based: one-stage](#content)
* **SSD: Single Shot MultiBox Detector.** ECCV, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1512.02325.pdf)]<br/>  
* **Focal Loss for Dense Object Detection.** ICCV, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1708.02002.pdf), [Code & pre-trained models](https://github.com/facebookresearch/detectron2)]<br/>
  
### [Anchor-Based: two-stage](#content)
* **Rich feature hierarchies for accurate object detection and semantic segmentation.** CVPR, 2014. <br/>
  [[Paper](https://arxiv.org/pdf/1311.2524.pdf)]<br/>
* **Fast R-CNN.** ICCV, 2015. <br/>
  [[Paper](https://arxiv.org/pdf/1504.08083.pdf), [Code & pre-trained models](https://github.com/rbgirshick/fast-rcnn)]<br/>
* **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.** NeurIPS, 2015. <br/>
  [[Paper](https://arxiv.org/pdf/1506.01497.pdf), [Code & pre-trained models](https://github.com/facebookresearch/detectron2)]<br/>
* **A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection.** ECCV, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1506.01497.pdf), [Code & pre-trained models](https://github.com/facebookresearch/detectron2)]<br/>
* **Training Region-based Object Detectors with Online Hard Example Mining.** CVPR, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1604.03540.pdf), [Codes](https://github.com/abhi2610/ohem), OHEM]<br/>
* **HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection.** CVPR, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1604.00600.pdf)]<br/>
* **R-FCN: Object Detection via Region-based Fully Convolutional Networks.** NeurIPS, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1605.06409.pdf), Code & pre-trained models: [Matlab (official)](https://github.com/daijifeng001/R-FCN), [Pytorch](https://github.com/YuwenXiong/py-R-FCN)]<br/>
* **Feature Pyramid Networks for Object Detection.** CVPR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1612.03144.pdf)]<br/>
* **A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection.** CVPR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1704.03414.pdf), [Code](https://github.com/xiaolonw/adversarial-frcnn)]<br/>
* **Cascade R-CNN: Delving into High Quality Object Detection.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1712.00726.pdf), [Extension](https://arxiv.org/pdf/1906.09756.pdf); Code & pre-trained models: [Caffe](https://github.com/zhaoweicai/cascade-rcnn), [Detectron-based](https://github.com/zhaoweicai/Detectron-Cascade-RCNN)]<br/>
* **An Analysis of Scale Invariance in Object Detection - SNIP.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1711.08189.pdf)]<br/>
* **Scale-Aware Trident Networks for Object Detection.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1901.01892.pdf), [Code & pre-trained models](https://github.com/TuSimple/simpledet/tree/master/models/tridentnet)]<br/>
* **Thundernet: Towards realtime generic object detection.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1903.11752.pdf)]<br/>
* **AutoFocus: Efficient Multi-Scale Inference.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1812.01600.pdf)]<br/>
* **Learning to Rank Proposals for Object Detection.** ICCV, 2019. <br/>
  [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.pdf)]<br/>
* **Bounding Box Regression with Uncertainty for Accurate Object Detection.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1809.08545.pdf), [Code](https://github.com/yihui-he/KL-Loss)]<br/>
  
### [Anchor-Free: Keypoint-based](#content)

### [Anchor-Free: Center-based](#content)


### [Transformers](#content)
* **End-to-End Object Detection with Transformers.** ECCV, 2020. <br/>
  [[Paper](https://arxiv.org/pdf/2005.12872.pdf)], [[Code & pre-trained models](https://github.com/facebookresearch/detr)]<br/>


## [Architecture](#content)
* **Bottleneck Transformers for Visual Recognition.** ECCV, 2020. <br/>
  [[Paper](https://arxiv.org/pdf/1912.11370.pdf), [Code & pre-trained models](https://github.com/google-research/big_transfer)]<br/>
  
  
## [Miscellaneous](#content)
* **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.** arXiv, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1706.02677.pdf)]<br/>
* **Rethinking ImageNet Pre-training.** arXiv, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1811.08883.pdf)]<br/>
* **Big Transfer (BiT): General Visual Representation Learning.** ECCV, 2020. <br/>
  [[Paper](https://arxiv.org/pdf/1912.11370.pdf), [Code & pre-trained models](https://github.com/google-research/big_transfer)]<br/>  
* **Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection.** CVPR, 2020. <br/>
  [[Paper](https://arxiv.org/pdf/1912.02424.pdf), [Code & pre-trained models](https://github.com/sfzhang15/ATSS)]<br/>
