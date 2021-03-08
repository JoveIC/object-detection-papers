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
<tr><td colspan="3"><a href="#architecture">2. Architecture </a></td></tr>
<tr><td colspan="3"><a href="#miscellaneous">2. Miscellaneous </a></td></tr>

</table>

## [Surveys](#content)
* **Deep Learning for Generic Object Detection: A Survey.** IJCV, 2019. <br/>
  [[Paper](https://link.springer.com/article/10.1007/s11263-019-01247-4)]<br/>  
* **A Survey of Deep Learning-based Object Detection.** IEEE Access, 2019. <br/>
  [[Paper](https://ieeexplore.ieee.org/document/8825470)]<br/> 
* **A survey on Image Data Augmentation for Deep Learning.** Journal of Big Data, 2019. <br/>
  [[Paper](https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-019-0197-0.pdf)]<br/>  
* **Transformers in Vision: A Survey.** arXiv, 2021. <br/>
  [[Paper](https://arxiv.org/pdf/2101.01169.pdf)]<br/>


## [Image Recognition](#content)
* **ImageNet Classification with Deep Convolutional Neural Networks.** NeurIPS, 2012. <br/>
  [[Paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), AlexNet]<br/>
* **Visualizing and Understanding Convolutional Networks.** ECCV, 2014. <br/>
  [[Paper](https://arxiv.org/pdf/1311.2901.pdf), ZFNet]<br/>
* **Caffe: Convolutional Architecture for Fast Feature Embedding.** ACM Multimedia, 2014. <br/>
  [[Paper](https://ucb-icsi-vision-group.github.io/caffe-paper/caffe.pdf), CaffeNet]<br/>
* **Very Deep Convolutional Networks for Large-Scale Image Recognition.** ICLR, 2015. <br/>
  [[Paper](https://arxiv.org/pdf/1409.1556.pdf), VGG]<br/>
* **Going deeper with convolutions.** CVPR, 2015. <br/>
  [[Paper](https://arxiv.org/pdf/1409.4842.pdf), GoogleNet(Inception V1)]<br/>
* **Rethinking the Inception Architecture for Computer Vision.** CVPR, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1512.00567.pdf), Inception V2/3]<br/>
* **Deep Residual Learning for Image Recognition.** CVPR, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1512.03385.pdf), ResNet]<br/>
* **SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.** CoRR, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1602.07360.pdf)]<br/>
* **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.** AAAI, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1602.07261.pdf), InceptionResNetV2]<br/>
* **Densely Connected Convolutional Networks.** CVPR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1608.06993.pdf), DenseNet]<br/>
* **Xception: Deep Learning with Depthwise Separable Convolutions.** CVPR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1610.02357.pdf)]<br/>
* **Aggregated Residual Transformations for Deep Neural Networks.** CVPR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1611.05431.pdf), [code](https://github.com/facebookresearch/ResNeXt), ResNeXt]<br/>
* **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.** CoRR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1704.04861.pdf)]<br/>
* **MobileNetV2: Inverted Residuals and Linear Bottlenecks.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1801.04381.pdf)]<br/>
* **Learning Transferable Architectures for Scalable Image Recognition.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1707.07012.pdf), NASNet]<br/>
* **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1707.01083.pdf)]<br/>
* **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.** ECCV, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1807.11164.pdf)]<br/>
* **Searching for MobileNetV3.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1905.02244.pdf)]<br/>
* **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1905.11946.pdf)]<br/>
* **MnasNet: Platform-Aware Neural Architecture Search for Mobile.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1807.11626.pdf)]<br/>

* **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** ICLR, 2021. <br/>
  [[Paper](https://arxiv.org/pdf/2010.11929.pdf), [Code & pre-trained models](https://github.com/google-research/vision_transformer)]<br/>


## [Object Detection](#content)
### [Anchor-Based: one-stage](#content)
* **SSD: Single Shot MultiBox Detector.** ECCV, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1512.02325.pdf)]<br/>  
* **Focal Loss for Dense Object Detection.** ICCV, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1708.02002.pdf), [Code & pre-trained models](https://github.com/facebookresearch/detectron2), RetinaNet]<br/>
* **DSSD : Deconvolutional Single Shot Detector.** CoRR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1701.06659.pdf)]<br/>
* **Single-Shot Refinement Neural Network for Object Detection.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1711.06897.pdf), [code](https://github.com/sfzhang15/RefineDet)]<br/> 
* **Single-Shot Object Detection with Enriched Semantics.** CVPR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1712.00433.pdf), [code](https://github.com/bairdzhang/des)]<br/>
* **Deep Feature Pyramid Reconfiguration for Object Detection.** ECCV, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1808.07993.pdf)]<br/> 
* **Receptive Field Block Net for Accurate and Fast Object Detection.** ECCV, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1711.07767.pdf), [code](https://github.com/ruinmessi/RFBNet)]<br/> 
* **ScratchDet: Training Single-Shot Object Detectors from Scratch.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1810.08425.pdf), [code](https://github.com/KimSoybean/ScratchDet)]<br/> 
* **Towards accurate one-stage object detection with ap-loss.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1904.06373.pdf), [Journal](https://arxiv.org/pdf/2008.07294.pdf); [code](https://github.com/cccorn/AP-loss)]<br/> 
* **FreeAnchor: Learning to Match Anchors for Visual Object Detection.** NeuraIPS, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1909.02466.pdf), [code](https://github.com/zhangxiaosong18/FreeAnchor)]<br/>  
* **Learning rich features at high-speed for single-shot object detection.** ICCV, 2019. <br/>
  [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Rich_Features_at_High-Speed_for_Single-Shot_Object_Detection_ICCV_2019_paper.pdf)]<br/>  
  
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
* **CornerNet: Detecting Objects as Paired Keypoints.** ECCV, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1808.01244.pdf), [code](https://github.com/princeton-vl/CornerNet)]<br/>
* **Grid R-CNN.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1811.12030.pdf), [code](https://github.com/STVIR/Grid-R-CNN)]<br/>
* **Bottom-up Object Detection by Grouping Extreme and Center Points.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1901.08043.pdf), [code](https://github.com/xingyizhou/ExtremeNet)]<br/>
* **CenterNet: Keypoint Triplets for Object Detection.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1904.08189.pdf), [code](https://github.com/Duankaiwen/CenterNet)]<br/>
* **RepPoints: Point Set Representation for Object Detection.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1904.11490.pdf), [code](https://github.com/microsoft/RepPoints)]<br/>
* **Objects as Points.** CoPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1904.07850.pdf), [code](https://github.com/xingyizhou/CenterNet)]<br/> 
* **CornerNet-Lite: Efficient Keypoint Based Object Detection.** BMVC, 2020. <br/>
  [[Paper](https://arxiv.org/pdf/1904.08900.pdf), [code](https://github.com/princeton-vl/CornerNet-Lite)]<br/>

### [Anchor-Free: Center-based](#content)
* **DenseBox: Unifying Landmark Localization with End to End Object Detection.** CoRR, 2015. <br/>
  [[Paper](https://arxiv.org/pdf/1509.04874.pdf)]<br/> 
* **You Only Look Once: Unified, Real-Time Object Detection.** CVPR, 2016. <br/>
  [[Paper](https://arxiv.org/pdf/1506.02640.pdf)]<br/>  
* **YOLO9000: Better, Faster, Stronger.** CVPR, 2017. <br/>
  [[Paper](https://arxiv.org/pdf/1612.08242.pdf), [Website](https://pjreddie.com/darknet/yolov2/)]<br/>  
* **YOLOv3: An Incremental Improvement.** CoRR, 2018. <br/>
  [[Paper](https://arxiv.org/pdf/1804.02767.pdf), [Website](https://pjreddie.com/darknet/yolo/)]<br/>
* **FCOS: Fully Convolutional One-Stage Object Detection.** ICCV, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1904.01355.pdf); [code](https://github.com/tianzhi0549/FCOS)]<br/>
* **Feature Selective Anchor-Free Module for Single-Shot Object Detection.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1903.00621.pdf)]<br/>  
* **Region Proposal by Guided Anchoring.** CVPR, 2019. <br/>
  [[Paper](https://arxiv.org/pdf/1901.03278.pdf); [code](https://github.com/open-mmlab/mmdetection)]<br/> 
* **YOLOv4: Optimal Speed and Accuracy of Object Detection.** CoRR, 2020. <br/>
  [[Paper](https://arxiv.org/pdf/2004.10934.pdf); [code](https://github.com/AlexeyAB/darknet)]<br/>

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
