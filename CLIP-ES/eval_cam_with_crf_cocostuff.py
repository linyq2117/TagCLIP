import torch
import os
import numpy as np
import torch.nn.functional as F
import joblib
import multiprocessing
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2
from PIL import Image
import argparse

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

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

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

def crf(n_jobs, is_coco=False):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # Process per sample
    def process(i):
        image_id = eval_list[i]
        image_path = os.path.join(args.image_root, image_id + '.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label_path = os.path.join(args.gt_root, image_id + '.png')
        gt_label = np.asarray(Image.open(label_path), dtype=np.int32)
        if is_cocostuff:
            gt_label = fine182_to_coarse[gt_label]
        # Mean subtraction
        image -= mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        filename = os.path.join(args.cam_out_dir, image_id + ".npy")
        cam_dict = np.load(filename, allow_pickle=True).item()
        cams = cam_dict['attn_highres']
        #bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        #cams = np.concatenate((bg_score, cams), axis=0)
        prob = cams

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)

        label = np.argmax(prob, axis=0)
        #keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        #label = keys[label]
        keys = np.array(cam_dict['keys'])
        label = keys[label]
        label = fine171_to_coarse[label]
        if not args.eval_only:
            confidence = np.max(prob, axis=0)
            #label[confidence < 0.95] = 255
            cv2.imwrite(os.path.join(args.pseudo_mask_save_path, image_id + '.png'), label.astype(np.uint8))

        return label.astype(np.uint8), gt_label.astype(np.uint8)

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
           [joblib.delayed(process)(i) for i in range(len(eval_list))]
    )
    if args.eval_only:
        preds, gts = zip(*results)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        #num_classes=21 if not is_coco else 81
        if is_cocostuff: num_classes = 27
        score = scores(gts, preds, n_class=num_classes)
        print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./cam_out", type=str)
    parser.add_argument("--pseudo_mask_save_path", default="", type=str)
    parser.add_argument("--split_file", default="/local_root/datasets/VOC2012/ImageSets/Segmentation/train.txt",
                        type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="/local_root/datasets/VOC2012/SegmentationClassAug", type=str)
    parser.add_argument("--image_root", default="/local_root/datasets/VOC2012/JPEGImages", type=str)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    #is_coco = 'coco' in args.cam_out_dir
    is_cocostuff = 'cocostuff' in args.cam_out_dir
    assert is_cocostuff
    #if 'voc' in args.cam_out_dir:
    #    eval_list = list(np.loadtxt(args.split_file, dtype=str))
    #elif 'coco' in args.cam_out_dir:
    file_list = tuple(open(args.split_file, "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    eval_list = [x[0] for x in file_list]#[:1000]
    print('{} images to eval'.format(len(eval_list)))

    if not args.eval_only and not os.path.exists(args.pseudo_mask_save_path):
        os.makedirs(args.pseudo_mask_save_path)

    mean_bgr = (104.008, 116.669, 122.675)
    n_jobs = 32 # multiprocessing.cpu_count()
    crf(n_jobs, is_coco)
