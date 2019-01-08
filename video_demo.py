from __future__ import print_function, absolute_import

import os

import cv2

import numpy as np
import imageio
import matplotlib.pyplot as plt
import torch
from pose_model import PoseModel
from torch.autograd import Variable
from collections import OrderedDict
from post import decode_pose

# Settings
mode = 'videos'
save_folder='results'
experiment_group = 'exp-4-epoch200'
experiment_name = 'multi-frame'
video_name = 'kunpeng_stand'
img_path = os.path.join('videos', video_name)
save_path = os.path.join('results', video_name, experiment_group, experiment_name)

if not os.path.isdir(save_path):
    os.makedirs(save_path, exist_ok=True)

# load pretrained model
if mode == 'videos':
    # weight_name = './checkpoint/multi-frame-epoch151.pth.tar'
    # weight_name = './checkpoint/multi-frame-epoch200.pth.tar'
    weight_name = './checkpoint/multi-frame-epoch200.pth.tar'
    single_frame = False
elif mode == 'images':
    # weight_name = './checkpoint/single-frame-epoch90.pth.tar'
    # weight_name = './checkpoint/multi-frame-epoch151.pth.tar'
    weight_name = './checkpoint/multi-frame-epoch200.pth.tar'
    single_frame = True
else:
    print('Mode is wrong!')


model = PoseModel(inp=3, keypoints=15, limbs=28).cuda()
checkpoint = torch.load(weight_name)

state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()



images = []
names = []
for root, _, fnames in sorted(os.walk(img_path)):
    for fname in sorted(fnames):     
        path1 = os.path.join(img_path, fname) 
        images.append(path1)
        names.append(fname)

for indx in range(len(images)):
    print('Processing..',names[indx])
    # resize image to suitable size
    image_original = imageio.imread(images[indx])
    h, w, _ = image_original.shape
    multiplier_in = 1.1
    multiplier = 1.0
    scale = 368./float(h) * multiplier
    image_original = cv2.resize(image_original, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image_original, (0,0), fx=multiplier_in, fy=multiplier_in, interpolation=cv2.INTER_CUBIC)

    image = image.astype(np.float32) / 255.

    # vgg preproess
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = image.copy()

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    preprocessed_img = torch.from_numpy(preprocessed_img).unsqueeze(0)

    image_input = Variable(preprocessed_img.cuda(), volatile=True)

    # inference heatmap
    if indx > 0:
        hms_est, pafs_est, hms_est_imp, pafs_est_imp = model(image_input, hms_est_imp, pafs_est_imp, single_frame)
    else:
        hms_est, pafs_est, hms_est_imp, pafs_est_imp = model(image_input, [], [], True)

                
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}

    hms = hms_est_imp.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
    pafs = pafs_est_imp.data.cpu().numpy().transpose(0, 2, 3, 1)[0]

    hms = cv2.resize(hms, (image_original.shape[1], image_original.shape[0]), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (image_original.shape[1], image_original.shape[0]), interpolation=cv2.INTER_CUBIC)

    # pose association
    canvas, to_plot, candidate, subset = decode_pose(image_original, param, hms, pafs)
    
    # save detected poses
    imageio.imwrite(os.path.join(save_path,names[indx]),to_plot)   



