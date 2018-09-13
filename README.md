# YOLOv3-Pytorch
PyTorch implementation of YOLOv3, including training and inference based on darknet and mobilnetv2
## Introduction
The YOLO is one of the most popular one stage object detector. In Mar 2018, [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), which is extremely fast and accurate has been released. The aim of this project is to replicate the [Darknet](https://github.com/pjreddie/darknet) implementation. It also supports training YOLOv3 network with various backends such as MobileNetv2, Darknet(53 or 21).


---
## Quick Start

## Training
1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify yolo_train.yml and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  

---
## Inference

## Todo list:
- [x] Training  
- [x] Multiscale-training
- [x] mobilnetv2 backends
- [ ] Multiscale-testing 
- [ ] soft-nms
- [ ] Multiple-GPU training
- [ ] mAP Evaluation
## Citation
- [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3), developed based on Keras + Numpy
- [BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch), Pytorch + Numpy, load pytorch pretrained model, loss does not converge now.
