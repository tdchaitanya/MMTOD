# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient, printnorm, printgradnorm

import sys
sys.path.insert(0, './lib/model/cgan')
from model.cgan.data import CreateDataLoader
from model.cgan.options.test_options import TestOptions
from model.cgan.models import create_model
from model.cgan.util.visualizer import save_images
from model.cgan.util import util
from copy import deepcopy

# from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet_dual import resnet
from collections import OrderedDict

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
  #cgan base options
  parser.add_argument('--dataroot', type=str, default='./data/VOCdevkit/VOC2007/JPEGImages',help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
  parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
  parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
  parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
  parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
  parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
  parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
  parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
  parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
  parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
  parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
  parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
  parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
  parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
  parser.add_argument('--dataset_mode', type=str, default='single', help='chooses how datasets are loaded. [unaligned | aligned | single]')
  parser.add_argument('--model', type=str, default='test',
                      help='chooses which model to use. cycle_gan, pix2pix, test')
  parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
  parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
  parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
  parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
  parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
  parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
  parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
  parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
  parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
  parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
  parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
  parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
  parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
  parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
  # cgan test options
  parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
  parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
  parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
  parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
  #  Dropout and Batchnorm has different behavioir during training and test.
  parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
  parser.add_argument('--num_test', type=int, default=1, help='how many test images to run')

  parser.set_defaults(model='cycle_gan')
  args = parser.parse_args()
  return args

def get_cgan_model():

  opt = TestOptions().parse()
  opt.num_threads = 1
  opt.batch_size =1
  opt.serial_batches = True
  opt.no_flip = True
  opt.display_id = -1
  opt.suffix = 'B'
  model = create_model(opt)
  model.setup(opt)
#   device = torch.device('cuda:{}'.format(0))
  net1, net2 = deepcopy(model.netG_A), deepcopy(model.netG_B)
  # net1 = net1.module
  net2 = net2.module

  return net2


class Resize_GPU(nn.Module):
    def __init__(self, h, w):
        super(Resize_GPU, self).__init__()
        self.op =  nn.AdaptiveAvgPool2d((h,w))
    def forward(self, x):
        x = self.op(x)
        return x

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  # if args.net == 'vgg16':
    # fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  if args.net == 'res101_cgan_update':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)

  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  cgan_b_to_a= get_cgan_model()
  device_1 = torch.device('cuda:{}'.format(1))

  cgan_b_to_a.load_state_dict(torch.load(f'{args.checkpoints_dir}/{args.name}/latest_net_G_B.pth', map_location=str(device_1)))

  # print(compare_models(cgan_a_to_b, cgan_b_to_a))
  for p in cgan_b_to_a.parameters(): p.requires_grad = False

  for key, p in cgan_b_to_a.named_parameters(): p.requires_grad = True

  def set_in_fix(m):
    classname = m.__class__.__name__
    if classname.find('InstanceNorm') != -1:
      for p in m.parameters(): p.requires_grad=False

  cgan_b_to_a.apply(set_in_fix)


  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  for key, value in dict(cgan_b_to_a.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  checkpoint_1 = torch.load('./models/res101_pascal/pascal_voc/faster_rcnn_1_15_662.pth')
  checkpoint_2 = torch.load('./models/res101_thermal/pascal_voc/faster_rcnn_1_15_1963.pth')

  checkpoint_1_model = OrderedDict([(k.replace('RCNN_base', 'RCNN_base_2'), v) for k, v in checkpoint_1['model'].items()  if 'RCNN_base' in k ])

  checkpoint_2_model = OrderedDict([(k.replace('RCNN_base', 'RCNN_base_1'), v) if 'RCNN_base' in k else (k, v) for k, v in checkpoint_2['model'].items()])

  checkpoint_2_model.update(checkpoint_1_model)

  checkpoint_2_model['RCNN_base_3.op.weight'] = fasterRCNN.state_dict()['RCNN_base_3.op.weight']
  checkpoint_2_model['RCNN_base_3.op.bias'] = fasterRCNN.state_dict()['RCNN_base_3.op.bias']

  fasterRCNN.load_state_dict(checkpoint_2_model)
  
  if 'pooling_mode' in checkpoint_2.keys():
      cfg.POOLING_MODE = checkpoint_2['pooling_mode']
  print("loaded rgb and thermal weights")
  
  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter(f'logs/{cfg.EXP_DIR}/')

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    cgan_b_to_a.train()

    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      im_shape = im_data.size()
      nw_resize = Resize_GPU(im_shape[2], im_shape[3])

      cgan_b_to_a.zero_grad()

      im_data_1ch = im_data.narrow(1, 0, 1)
      
      im_data_1 = cgan_b_to_a(im_data_1ch)
      im_data_1 = nw_resize(im_data_1)

      # print(f'Done')
      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data_1, im_data, im_info, gt_boxes, num_boxes)
            
      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      # f_loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          # loss_feat = f_loss.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          # loss_feat = f_loss.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        # print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f, feat_loss %.4f" \
        #               % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_feat))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f " \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            # 'loss_feat': loss_feat
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          import torchvision.utils as vutils
          x1 = vutils.make_grid(im_data, normalize=True, scale_each=True)
          logger.add_image("images_s_{}/original_thermal_image".format(args.session), x1, (epoch - 1) * iters_per_epoch + step)
          # logger.add_images("images_s_{}/frcnn_input_image".format(args.session), im_data, (epoch - 1) * iters_per_epoch + step)
          x3 = vutils.make_grid(im_data_1ch, normalize=True, scale_each=True)
          logger.add_image("images_s_{}/1ch_thermal".format(args.session), x3, (epoch - 1) * iters_per_epoch + step)

          x2 = vutils.make_grid(im_data_1, normalize=True, scale_each=True)
          logger.add_image("images_s_{}/generated_rgb".format(args.session), x2, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_name_b_2_a = os.path.join(output_dir, 'cgan_b_to_a_{}_{}_{}.pth'.format(args.session, epoch, step))

    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)

    save_checkpoint({
     'model': cgan_b_to_a.state_dict(),
    }, save_name_b_2_a)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
