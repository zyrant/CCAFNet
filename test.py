import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from rgbd.rgbd_models.CCAFNet import CCAFNet
from config import opt
from rgbd.rgbd_dataset import  test_dataset
from torch.cuda import amp


dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = CCAFNet()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
# model.load_state_dict(torch.load('/media/zy/shuju/TMMweight/TMMALLCFM/TMM_epoch_100.pth'))
model.load_state_dict(torch.load('/media/zy/shuju/RGBDweight/PVTbackbone_SC/II_epoch_best.pth'))

# model.load_state_dict(torch.load('/media/zy/shuju/TMMweight/vgg16plus/TMM_epoch_60.pth'))
model.cuda()
model.eval()

#test
test_mae = []
test_datasets = ['NJU2K','STERE','DES','LFSD','NLPR','SIP']

for dataset in test_datasets:
    mae_sum  = 0
    save_path = '/home/zy/PycharmProjects/SOD/rgbd/rgbd_test_maps/CCAFNet/' + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    for i in range(test_loader.size):
        image,  gt, depth, name  = test_loader.load_data()
        gt = gt.cuda()
        image = image.cuda()
        # print(image.shape)
        n, c, h, w = image.size()
        depth = depth.cuda()
        depth = depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)
        res  = model(image, depth)
        predict = torch.sigmoid(res)
        predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
        mae = torch.sum(torch.abs(predict - gt)) / torch.numel(gt)
        mae_sum = mae.item() + mae_sum
        predict = predict.data.cpu().numpy().squeeze()
        print('save img to: ', save_path + name)

        plt.imsave(save_path + name, arr=predict, cmap='gray')

    test_mae.append(mae_sum / test_loader.size)
print('Test_mae:', test_mae)
print('Test Done!')
