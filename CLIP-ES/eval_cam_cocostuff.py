
import numpy as np
import os
from PIL import Image
import argparse
import cv2

fine_to_coarse_dic = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}
fine182_to_coarse = np.array(list(fine_to_coarse_dic.values())+[255]*(256-182))
coco_stuff_171_to_27 = {}
cnt = 0
for fine, coarse in fine_to_coarse_dic.items():
    if fine + 1 in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:  # note that +1 is added
        continue
    coco_stuff_171_to_27[cnt] = coarse
    cnt += 1
fine171_to_coarse = np.array(list(coco_stuff_171_to_27.values())+[255]*(256-171))

def print_iou(iou, dname='voc'):
    iou_dict = {}
    for i in range(len(iou)-1):
        iou_dict[i] = iou[i+1]
    print(iou_dict)

    return iou_dict

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def run_eval_cam(args, print_log=False, is_coco=False, is_cocostuff=False):
    preds = []
    labels = []
    n_images = 0
    for i, id in enumerate(eval_list):
        n_images += 1
        if args.cam_type == 'png':
            label_path = os.path.join(args.cam_out_dir, id + '.png')
            cls_labels = np.asarray(Image.open(label_path), dtype=np.uint8)
            if is_cocostuff:
                cls_labels = fine171_to_coarse[cls_labels]
        else:
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict[args.cam_type]
            if 'bg' not in args.cam_type and not is_cocostuff:
                if args.cam_eval_thres < 1:
                    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
                else:
                    bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), args.cam_eval_thres)
                    cams = np.concatenate((bg_score, cams), axis=0)
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            if not is_cocostuff:
                cls_labels = keys[cls_labels]#.astype(np.uint8)
            else:
                keys = np.array(cam_dict['keys'])
                cls_labels = keys[cls_labels]
                cls_labels = fine171_to_coarse[cls_labels]
                
            #cv2.imwrite(os.path.join(args.cam_out_dir+'_crf', image_id + '.png'), cls_labels.astype(np.uint8))
        preds.append(cls_labels.astype(np.uint8))
        gt_file = os.path.join(args.gt_root, '%s.png' % id)
        gt = np.array(Image.open(gt_file))#.astype(np.uint8)
        if np.max(gt)>171 and np.max(gt)<255:
            print(np.max(gt))
        if is_cocostuff:
            gt = fine182_to_coarse[gt]
        labels.append(gt.astype(np.uint8))

    num_classes = 21 if not is_coco else 81
    if is_cocostuff: num_classes = 27
    iou = scores(labels, preds, n_class=num_classes)

    if print_log:
        print(iou)

    return iou["Mean IoU"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./cam_out", type=str)
    parser.add_argument("--cam_type", default="attn_highres", type=str)
    parser.add_argument("--split_file", default="/home/xxx/datasets/VOC2012/ImageSets/Segmentation/train.txt", type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="/home/xxx/datasets/VOC2012/SegmentationClassAug", type=str)
    args = parser.parse_args()

    is_coco = 'coco' in args.cam_out_dir
    is_cocostuff = 'cocostuff' in args.cam_out_dir
    #if 'voc' in args.cam_out_dir:
    #    eval_list = list(np.loadtxt(args.split_file, dtype=str))
    #elif 'coco' in args.cam_out_dir:
    file_list = tuple(open(args.split_file, "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    eval_list = [x[0] for x in file_list]#[:1000]
    print('>>>>evaluating {}'.format(args.cam_out_dir.split('/')[-1]))
    print('{} images to eval'.format(len(eval_list)))

    if 'bg' in args.cam_type or 'png' in args.cam_type:
        iou = run_eval_cam(args, True, is_coco=is_coco, is_cocostuff=is_cocostuff)
    else:
        if args.cam_eval_thres < 1:
            thres_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        else:
            if 'attn' in args.cam_type:
                thres_list = [1]
            else:
                thres_list =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_iou = 0
        max_thres = 0
        for thres in thres_list:
            args.cam_eval_thres = thres
            iou = run_eval_cam(args, print_log=False, is_coco=is_coco, is_cocostuff=is_cocostuff)
            print('thres:{}, mIoU:{}'.format(thres, iou))
            if iou > max_iou:
                max_iou = iou
                max_thres = thres

        args.cam_eval_thres = max_thres
        iou = run_eval_cam(args, print_log=True, is_coco=is_coco, is_cocostuff=is_cocostuff)
        print('\n')
