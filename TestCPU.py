import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
import cv2 as cv
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
parser.add_argument('--save_all', type=bool, default=True)
opt = parser.parse_args()

model = SINet_ResNet50().cpu()
model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')))
model.eval()

close = False
for dataset in ['COD10K-v3', 'MYTEST', 'CAMO', 'CHAMELEON', 'COD10K']:
    if close:
        break
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    imgpath = './Dataset/TestDataset/{}/Imgs/'.format(dataset)
    gtpath = './Dataset/TestDataset/{}/GT/'.format(dataset)
    if dataset == 'COD10K-v3':
        imgpath = './Dataset/TestDataset/{}/Image/'.format(dataset)
        gtpath = './Dataset/TestDataset/{}/GT_Instance/'.format(dataset)
    test_loader = test_dataset(imgpath, gtpath, opt.testsize)

    img_count = 1
    for iteration in range(test_loader.size):
        start_time = time.time()
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        _, cam = model(image)
        cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # convert to numpy array
        gt = (gt * 255).astype(np.uint8)
        _, gt_bw = cv.threshold(gt, 1, 255, cv.THRESH_BINARY)
        cam = (cam * 255).astype(np.uint8)
        _, cam_bw = cv.threshold(cam, 1, 255, cv.THRESH_BINARY)

        # dice index for scoring
        tp = cv.countNonZero(cv.bitwise_and(gt_bw, cam_bw))
        fp = cv.countNonZero(cv.bitwise_and(~gt_bw, cam_bw))
        fn = cv.countNonZero(cv.bitwise_and(gt_bw, ~cam_bw))
        dice = 2 * tp / (2 * tp + fp + fn)

        # convert original image to numpy array
        img = F.interpolate(image, size=gt.shape, mode='bilinear', align_corners=True)
        img = np.transpose(img.sigmoid().data.cpu().squeeze().numpy(), (1,2,0))
        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC3)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # highlight object in image
        gt_locs = np.where(gt_bw != 0)
        obj = img // 3
        obj[gt_locs[0], gt_locs[1]] = img[gt_locs[0], gt_locs[1]]
        cam_locs = np.where(cam_bw != 0)
        res = img // 3
        res[cam_locs[0], cam_locs[1]] = img[cam_locs[0], cam_locs[1]]

        duration = time.time() - start_time
        print('({}/{}) Dice: {:4.1f}%, {:.0f} ms, Image: {}/{}'.format(
            img_count, test_loader.size, 100 * dice, 1000 * duration, dataset, name),
            flush=True)

        ch = ord('s')
        if not opt.save_all:
            #cv.imshow('original', img)
            #cv.imshow('object', obj)
            cv.imshow('detected', res)
            ch = cv.waitKey() & 0xFF
            if ch is 27:    # ESC is pressed
                break
        if chr(ch).lower() == 's':
            res = cv.imwrite(save_path + name, cam)
            if res:
                print('Wrote out %s' % save_path + name)
            else:
                print('Could not write to %s' % save_path + name)

        img_count += 1

print("\n[Congratulations! Testing Done]")
