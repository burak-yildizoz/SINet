import os
import numpy as np
import cv2 as cv

result_root = './Result/2020-CVPR-SINet-New/'

close = False
for dataset in [ 'MYTEST', 'CHAMELEON', 'COD10K', 'CAMO', 'COD10K-v3' ]:
    if close:
        break
    result_path = result_root + dataset + '/'
    images = [f for f in os.listdir(result_path) if f.endswith('.png')]
    orig_path = './Dataset/TestDataset/' + dataset + '/Imgs/'
    gt_path = './Dataset/TestDataset/' + dataset + '/GT/'
    if dataset is 'COD10K-v3':
        images = [f for f in images if 'Deer' in f]
        orig_path = orig_path.replace('/Imgs/', '/Test/Image/')
        gt_path = gt_path.replace('/GT/', '/Test/GT_Object/')

    img_count = 1
    for file in images:
        file = file.replace('.png', '')

        mask = cv.imread(result_path + file + '.png', cv.IMREAD_GRAYSCALE)
        mask = np.where(mask > 0, 255, 0)

        gt = cv.imread(gt_path + file + '.png', cv.IMREAD_GRAYSCALE)
        gt = np.where(gt > 0, 255, 0)

        orig = cv.imread(orig_path + file + '.jpg', cv.IMREAD_COLOR)

        # dice index for scoring
        tp = cv.countNonZero(cv.bitwise_and(gt, mask))
        fp = cv.countNonZero(cv.bitwise_and(~gt, mask))
        fn = cv.countNonZero(cv.bitwise_and(gt, ~mask))
        dice = 2 * tp / (2 * tp + fp + fn)

        # highlight object in image
        gt_locs = np.where(gt != 0)
        obj = orig // 3
        obj[gt_locs[0], gt_locs[1]] = orig[gt_locs[0], gt_locs[1]]
        cam_locs = np.where(mask != 0)
        res = orig // 3
        res[cam_locs[0], cam_locs[1]] = orig[cam_locs[0], cam_locs[1]]

        print('({}/{}) Dice: {:4.1f}%, Image: {}/{}'.format(
            img_count, len(images), 100 * dice, dataset, file),
            flush=True)

        cv.imshow('original', orig)
        cv.imshow('object', obj)
        cv.imshow('detected', res)
        ch = cv.waitKey() & 0xFF
        if ch is 27:    # ESC is pressed
            close = True
            break
        if chr(ch).lower() == 's':
            cv.imwrite('./Result/' + file + '_original.jpg', orig)
            cv.imwrite('./Result/' + file + '_object.jpg', obj)
            cv.imwrite('./Result/' + file + '_detected.jpg', res)
            print('Wrote out results for ' + file, flush=True)
        img_count += 1
