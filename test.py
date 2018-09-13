import torch

# losses_name = ["loss", "xy", "wh", "conf", "clss", "nRecall50", "nRecall75", "nProposal"]
# losses = [[],[],]
# for i, name in enumerate(losses_name):
#     logger.add_scalar('DET_', losses[i].sum().item() / losses[i].numel(), step)
# losses = torch.randn(1, 3)
# print(losses)
# b = losses.unsqueeze(0)
# print(b)
# print(losses[0].numel())
# print(losses[0].sum().item() / losses[0].numel())

# a = torch.Tensor(2,3)
# print(a)
# b = a.view(1, -1)
# print(b)
# i = 0
# n = 10000
# for x in range(1000):
#     for j in range(10):
#         i = (i+1) % n
#
#         print(i)
# print(int(13/2))
#
#
# def test(i=0):
#     while 1:
#         i =i + 1
#         yield i + 1
#
#
# a = test()
# # b = iter(a)
# print(next(a))
# print(next(a))
# print(next(a))
# print(next(a))
# import argparse
# import yaml
#
# parser = argparse.ArgumentParser()
# # # Data input settings
# parser.add_argument('--opt_path', type=str, default='cfgs/coco_det.yml', help='')
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether use gpu.')
# parser.add_argument('--mGPUs', type=bool, default=False, help='whether use mgpu.')
#
#
# def update_values(dict_from, dict_to):
#     for key, value in dict_from.items():
#         if isinstance(value, dict):
#             update_values(dict_from[key], dict_to[key])
#         elif value is not None:
#             dict_to[key] = dict_from[key]
#
#
# opt = parser.parse_args()
# with open(opt.opt_path, 'r') as handle:
#     options_yaml = yaml.load(handle)
# print(options_yaml)
# update_values(options_yaml, vars(opt))
#
# # if opt.seed:
# #     torch.manual_seed(opt.seed)
# #     if opt.use_cuda:
# #         torch.cuda.manual_seed(opt.seed)
#
# print(opt)
import signal
import sys
import time


# def signal_handler(signal, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
#
#
# def signal_handler2(signal, frame):
#     print('Ctrl+C!')
#     sys.exit(0)
#
#
# signal.signal(signal.SIGINT, signal_handler)
#
# if __name__ == '__main__':
#
#     signal.signal(signal.SIGINT, signal_handler2)
#     try:
#         for x in range(1, 500):
#             time.sleep(0.2)
#             print(x)
#     except KeyboardInterrupt:
#         print("xx")
# import datetime
# print(time.strftime("%Y%m%d%H%M%S", time.localtime()))
# print(str(datetime.datetime.now()) + '_' + '_' + 'bs:' + str(1))

# with open('F:/Self-driving_Object_Detection/projs/YOLOv3_Pytorchv2/cfgs/20180712_train_Sign_Veh_crop.txt') as f:
#     lines = f.readlines()
# with open('xxx.txt', 'w') as fr:
#     for line in lines:
#         linea = line.split(' ')
#         fr.write(linea[0][0:12]+"20180712_Vechicle&TS_annotation" + line[11:])

# a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# b = torch.Tensor([[10,20,30],[40,50,6],[70,80,90]])
# c = torch.Tensor([[100,200,300],[400,500,600],[700,800,900]])
# print(a)
# print(b)
# print(c)
# d = torch.stack((a,b,c),dim=1)
# print(d)
#
# feats = torch.randn(416, 416, 3, 4)
# #feats = feats.cuda()
# start = time.time()
# for i in range(1000):
#     feats = feats.view(-1, 3, 4, 416, 416).permute(0, 3, 4, 1, 2).contiguous()
# print(time.time() - start)

# feats = torch.randn(4, 416, 416, 3)
# #x = feats.unsqueeze(0)
# img = feats.view(feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3]).permute(0, 3, 1, 2).contiguous()
# print(img.shape)
# import numpy as np
#
#
# def box_iou(b1, b2):
#     '''Return iou tensor
#
#     Parameters
#     ----------
#     b1: tensor, shape=(i1,...,iN, 4), xywh
#     b2: tensor, shape=(j, 4), xywh
#
#     Returns
#     -------
#     iou: tensor, shape=(i1,...,iN, j)
#
#     '''
#
#     # Expand dim to apply broadcasting.
#     b1 = b1.unsqueeze(3)
#
#     b1_xy = b1[..., :2]
#     b1_wh = b1[..., 2:4]
#     b1_wh_half = b1_wh / 2.
#     b1_mins = b1_xy - b1_wh_half
#     b1_maxes = b1_xy + b1_wh_half
#
#     # if b2 is an empty tensor: then iou is empty
#     if b2.shape[0] == 0:
#         iou = torch.zeros(b1.shape[0:4]).type_as(b1)
#     else:
#         b2 = b2.view(1, 1, 1, b2.size(0), b2.size(1))
#         # Expand dim to apply broadcasting.
#         b2_xy = b2[..., :2]
#         b2_wh = b2[..., 2:4]
#         b2_wh_half = b2_wh / 2.
#         b2_mins = b2_xy - b2_wh_half
#         b2_maxes = b2_xy + b2_wh_half
#
#         intersect_mins = torch.max(b1_mins, b2_mins)
#         intersect_maxes = torch.min(b1_maxes, b2_maxes)
#         intersect_wh = torch.clamp(intersect_maxes - intersect_mins, min=0)
#
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         b1_area = b1_wh[..., 0] * b1_wh[..., 1]
#         b2_area = b2_wh[..., 0] * b2_wh[..., 1]
#         iou = intersect_area / (b1_area + b2_area - intersect_area)
#
#     return iou
#
#
# x = np.array([[ 181.7490,  284.3782,  196.0428,  302.9306,    0.9965],
#         [ 181.3222,  284.0953,  196.4672,  303.3017,    0.9902],
#         [ 181.4702,  235.3388,  196.7543,  256.3076,    0.9825],
#         [ 182.2063,  260.5985,  195.5836,  278.6449,    0.9805],
#         [ 182.6984,  236.5589,  195.6154,  254.8922,    0.9722],
#         [ 176.9854,  219.5637,  184.2798,  237.2380,    0.9458],
#         [ 182.2640,  219.9757,  196.5678,  236.7201,    0.9416],
#         [ 180.9963,  218.4672,  197.2836,  238.4083,    0.9354],
#         [ 181.5333,  259.8358,  196.2119,  279.4641,    0.9353],
#         [ 202.9288,  265.5182,  206.1805,  270.7218,    0.8281],
#         [ 181.6683,  281.6216,  195.8764,  305.5264,    0.8165],
#         [ 182.5962,  243.3247,  195.9495,  261.2517,    0.8052],
#         [ 181.8593,  234.4378,  196.4199,  256.8516,    0.7250],
#         [ 181.8905,  241.8454,  196.2685,  262.5981,    0.6897],
#         [ 201.5233,  255.6402,  209.3316,  267.1790,    0.5835],
#         [ 181.7884,  217.8391,  196.3681,  238.8092,    0.5130]])
#
#
# def py_nms(dets, thresh):
#     """Pure Python NMS baseline."""
#     #x1、y1、x2、y2、以及score赋值
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]
#
#     #每一个候选框的面积
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     #order是按照score降序排序的
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         #计算当前概率最大矩形框与其他矩形框的相交框的坐标
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         #计算相交框的面积
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         #找到重叠度不高于阈值的矩形框索引
#         inds = np.where(ovr <= thresh)[0]
#         #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
#         order = order[inds + 1]
#     return keep
#
# time1 = time.time()
# for i in range(100):
#     py_nms(x, 0.3)
# # print(time.time()-time1)\
# feats = torch.rand(10, 33, 13, 13)
# grid_shape = (feats.shape[2:4])
# grid_y = torch.arange(0, grid_shape[0]).view(-1, 1, 1, 1).expand(grid_shape[0], grid_shape[0], 1, 1)
# grid_x = torch.arange(0, grid_shape[1]).view(1, -1, 1, 1).expand(grid_shape[1], grid_shape[1], 1, 1)
# import math
# import matplotlib.pyplot as plt
#
# init_lr = 0.0001
# a = []
# epoch_start = 0
# epoch_max = 20
# step = 5
# max_epoch =100
# for epoch in range(max_epoch):
#     if epoch % (max_epoch/step) == 0:
#         epoch_max = 20
#         epoch_start = 0
#     curr_lr = init_lr*(1+math.cos((epoch_start/epoch_max)*math.pi))/2
#     epoch_start += 1
#     a.append(curr_lr)
# b = [i for i in range(100)]
# print(a)
# plt.plot(b, a)
# plt.show()
x = []
a = torch.Tensor(x)
torch.cat(x, dim=0)
