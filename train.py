import argparse
import numpy as np
import math
import os
import time
from six.moves import cPickle  # six.moves is used to self-adjust the change of python 2 and python 3
import yaml
import itertools
from multiprocessing.dummy import Pool as ThreadPool

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from model import yolo_layer
from utils.utils import get_random_data, preprocess_true_boxes, add_logger, update_values, set_tb_logger


class Train(object):
    def __init__(self, args, threadpool):

        # initial parameters and model
        self.args = args
        self.annotation_path = args.annotation_path
        self.input_shape = (args.height, args.width)

        self.anchors = args.anchors
        self.class_names = args.classes_names
        self.num_classes = len(self.class_names)
        self.model, self.infos = self._create_model()
        self.optimizer = self._get_optimizer(self.args, self.model)
        self.threadpool = threadpool

        # mkdir log file
        log_name = str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + '_' + '_' + 'bs_' + str(self.args.batch_size)
        print("logging to %s ..." % (log_name))
        self.logger = set_tb_logger('logs', log_name)
        self._train_pipeline(self.annotation_path)
        self.init_lr = args.lr

    def _train_pipeline(self, sample_path):
        '''
            The training pipeline including training and validation samples generation, loss backward
            and checkpoints saving.
        '''
        # random split the data to training and validation samples
        data_gen, data_gen_val, max_batch_ind, max_val_batch_ind = self._train_data_generation(sample_path)

        self.args.n_iter = self.infos.get('n_iter', self.args.n_iter)
        start_epoch = self.infos.get('epoch', self.args.start_epoch)
        best_val_loss = self.infos.get('best_val_loss', 100000)

        # Learning rate self-adjusting depends on the validation loss
        scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=2, verbose=True, eps=1e-12)
        # scheduler = StepLR(self.optimizer, step_size=3, gamma=0.6)

        # training approach
        for epoch in range(start_epoch, self.args.max_epochs):
            self._train(data_gen, max_batch_ind, epoch)
            val_loss = self._validation(epoch, data_gen_val, max_val_batch_ind)
            scheduler.step(val_loss)
            best_flag = False

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_flag = True

            # checkpoints saving
            self.checkpoint_save(self.infos, epoch, best_val_loss, best_flag)

    def cos_lr(self, epoch_start, epoch_max, base_lr):
        curr_lr = base_lr * (1 + math.cos(math.pi*epoch_start / epoch_max)) / 2
        return curr_lr

    def _train(self, data_gen, max_batch_ind, epoch):
        '''
        :param data_gen: generator for training data
        :param max_batch_ind: maximum slice for training data
        :param epoch:
        '''

        torch.set_grad_enabled(True)
        self.model.train()
        tmp_losses = 0
        batch_sum = 0
        start = time.time()
        for batch_ind in range(max_batch_ind):
            data = next(data_gen)
            img, label0, label1, label2 = data[0], data[1], data[2], data[3]
            img = torch.from_numpy(img).float().cuda()
            img = img.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).permute(0, 3, 1, 2).contiguous()
            label0 = torch.from_numpy(label0)
            label1 = torch.from_numpy(label1)
            label2 = torch.from_numpy(label2)

            if self.args.use_cuda:
                img = img.cuda()
                label0 = label0.cuda()
                label1 = label1.cuda()
                label2 = label2.cuda()

            losses = self.model(img, label0, label1, label2)
            loss = add_logger(losses, self.logger, batch_ind + max_batch_ind * epoch, 'train')
            loss = loss.sum() / loss.numel()
            tmp_losses = tmp_losses + loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_ind % self.args.display_interval == 0 and batch_ind != 0:
                end = time.time()
                batch_sum = batch_sum + self.args.display_interval
                tmp_losses_show = tmp_losses / batch_sum
                print("step {}/{} (epoch {}), loss: {:f} , lr:{:f}, time/batch = {:.3f}"
                      .format(batch_ind, max_batch_ind, epoch, tmp_losses_show, self.optimizer.param_groups[-1]['lr'],
                              (end - start)/self.args.display_interval))
                start = end

    def _validation(self, epoch, data_gen, max_val_batch_ind):
        torch.set_grad_enabled(False)
        self.model.eval()
        tmp_losses = 0

        for batch_idx in range(max_val_batch_ind):
            data = next(data_gen)
            img, label0, label1, label2 = data[0], data[1], data[2], data[3]
            img = torch.from_numpy(img).float()
            img = img.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).permute(0, 3, 1, 2).contiguous()

            label0 = torch.from_numpy(label0)
            label1 = torch.from_numpy(label1)
            label2 = torch.from_numpy(label2)

            if self.args.use_cuda:
                img = img.cuda()
                label0 = label0.cuda()
                label1 = label1.cuda()
                label2 = label2.cuda()

            losses = self.model(img, label0, label1, label2)
            loss = add_logger(losses, self.logger, self.args.n_iter, 'val')
            tmp_losses = tmp_losses + loss.item()

        tmp_losses = tmp_losses / max_val_batch_ind
        print("============================================")
        print("Evaluation Loss (epoch {}), TOTAL_LOSS: {:.3f}".format(epoch, tmp_losses))
        print("============================================")
        return tmp_losses

    @staticmethod
    def _get_optimizer(args, net):
        params = []
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'backbone' in key:
                    params += [{'params':[value], 'lr':args.backbone_lr}]
                else:
                    params += [{'params':[value], 'lr':args.lr}]

        # Initialize optimizer class
        if args.optimizer == "adam":
            optimizer = optim.Adam(params, weight_decay=args.weight_decay)
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(params, weight_decay=args.weight_decay)
        else:
            # Default to sgd
            optimizer = optim.SGD(params, momentum=0.9, weight_decay=args.weight_decay,
                                  nesterov=(args.optimizer == "nesterov"))
        return optimizer

    def _create_model(self):
        model = yolo_layer.yolov3layer(self.args)
        infos = {}
        if self.args.start_from != '':
            if self.args.load_best_score == 1:
                model_path = os.path.join(self.args.start_from, 'model-best.pth')
                info_path = os.path.join(self.args.start_from, 'infos-best.pkl')
            else:
                model_path = os.path.join(self.args.start_from, 'model.pth')
                info_path = os.path.join(self.args.start_from, 'infos.pkl')

            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    infos = cPickle.load(f)

            print('Loading the model from %s ...' %(model_path))
            model.load_state_dict(torch.load(model_path))

        if self.args.use_cuda:
            if self.args.mGPUs:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        return model, infos

    def _data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            if i + batch_size > n:
                np.random.shuffle(annotation_lines)
                i = 0

            # Separate all image reader into different threads can speed up the program
            output = self.threadpool.starmap(get_random_data, zip(annotation_lines[i:i + batch_size],
                                                                  itertools.repeat(input_shape, batch_size)))
            image_data = list(zip(*output))[0]
            box_data = list(zip(*output))[1]
            i = i + batch_size
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)

            yield [image_data, *y_true]

    def _data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return self._data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    def _train_data_generation(self, sample_path):

        with open(sample_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        val_split = self.args.val_split
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        batch_size = self.args.batch_size
        data_gen = self._data_generator_wrapper(lines[:num_train], batch_size, self.input_shape, self.anchors, self.num_classes)
        data_gen_val = self._data_generator_wrapper(lines[num_train:], batch_size, self.input_shape, self.anchors, self.num_classes)
        max_batch_ind = int(num_train / batch_size)
        max_val_batch_ind = int(num_val / batch_size)

        return data_gen, data_gen_val, max_batch_ind, max_val_batch_ind

    def checkpoint_save(self, infos, epoch, best_val_loss, best_flag=False):

        checkpoint_path = os.path.join(self.args.save_path, self.args.backbones_network)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if self.args.mGPUs > 1:
            torch.save(self.model.module.state_dict(), os.path.join(checkpoint_path, 'model.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, 'model.pth'))

        print("model saved to {}".format(checkpoint_path))
        infos['n_iter'] = self.args.n_iter
        infos['epoch'] = epoch
        infos['best_val_loss'] = best_val_loss
        infos['opt'] = self.args

        with open(os.path.join(checkpoint_path, 'infos.pkl'), 'wb') as f:
            cPickle.dump(infos, f)

        if best_flag:
            if self.args.mGPUs > 1:
                torch.save(self.model.module.state_dict(), os.path.join(checkpoint_path, 'model-best.pth'))
            else:
                torch.save(self.model.state_dict(), os.path.join(checkpoint_path, 'model-best.pth'))

            print("model saved to {} with best total loss {:.3f}".format(os.path.join(checkpoint_path, \
                                                                                      'model-best.pth'), best_val_loss))

            with open(os.path.join(checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                cPickle.dump(infos, f)


def main():
    args = make_args()
    with open(args.cfg_path, 'r') as handle:
        options_yaml = yaml.load(handle)
    update_values(options_yaml, vars(args))

    # set random seed to cpu and gpu
    if args.seed:
        torch.manual_seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)

    try:
        threadpool = ThreadPool(args.batch_size)
    except Exception as e:
        print(e)
        exit(1)

    Train(args, threadpool)


def make_args():
    # load the optional parameters and update new arguments
    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--cfg_path', type=str, default='cfgs/Yolo_train.yml', help='load config')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether use gpu.')
    parser.add_argument('--mGPUs', type=bool, default=False, help='whether use mgpu.')
    return parser.parse_args()


if __name__ == '__main__':

    main()

