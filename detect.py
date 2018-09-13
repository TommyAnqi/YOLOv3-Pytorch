import torch
import colorsys
import numpy as np
import argparse
import yaml
from utils.utils import  update_values, letterbox_image
from model import yolo_layer
import time
import cv2


class yolo_detect(object):
    def __init__(self, args):
        self.args = args
        self.model_image_size = (args.height, args.width)
        self.class_names = self.args.classes_names
        self.model = self.creat_model()

    def creat_model(self):
        model_path = self.args.inference_path
        model = yolo_layer.yolov3layer(self.args)
        model.load_state_dict(torch.load(model_path))
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        if self.args.use_cuda:
            if self.args.mGPUs:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        return model

    def detect(self, image):
        time_crop_img_s = time.time()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
        time_crop_img_e = time.time()
        print('Time consuming of image crop:', time_crop_img_e - time_crop_img_s)
        time_inference_s = time.time()
        image_data = np.array(boxed_image, dtype='float32')
        img = torch.from_numpy(image_data).float().cuda()
        img = img.unsqueeze(0)
        img = img.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).permute(0, 3, 1, 2).contiguous()
        img_ori_shape = torch.Tensor([image.shape[0], image.shape[1]])
        dets, images, classes = self.model.detect(img, img_ori_shape)
        time_inference_e = time.time()
        print('Time consuming of inference:', time_inference_e - time_inference_s)
        dets_arr = dets.cpu().numpy()
        classes_arr = classes.cpu().numpy()
        time_draw_s = time.time()
        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #             size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        for i, c in enumerate(classes_arr):
            c = int(c)
            predicted_class = self.class_names[c]
            box = dets_arr[i]

            top, left, bottom, right, score = box
            label = '{} {:.2f}'.format(predicted_class, score)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], 2)
        time_draw_e = time.time()
        print('Time consuming of Drawing image:', time_draw_e - time_draw_s)
        return image


def detect_img_test(yolo):
    while 1:
        image = input("input image:")
        try:
            time_load_start = time.time()
            image = cv2.imread(image)
            time_load_end = time.time()
            print('Time consuming of image loading:', time_load_end - time_load_start)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect(image)
            time_end = time.time()
            print('Time consuming of object detection based on yolo3', time_end - time_load_start)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow('result', r_image)
            cv2.waitKey(0)


def main():
    args = make_args()
    with open(args.cfg_path, 'r') as handle:
        options_yaml = yaml.load(handle)
    update_values(options_yaml, vars(args))
    detect_img_test(yolo_detect(args))


def make_args():
    # load the optional parameters and update new arguments
    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--cfg_path', type=str, default='cfgs/Yolo_detect.yml', help='load config')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether use gpu.')
    parser.add_argument('--mGPUs', type=bool, default=False, help='whether use mgpu.')
    return parser.parse_args()


if __name__ == '__main__':
    main()


