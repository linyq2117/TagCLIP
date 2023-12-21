import clip
import torch
import cv2
import numpy as np
from PIL import Image

import pickle
import os
from tqdm import tqdm
from lxml import etree
import math

import torch.nn.functional as F
import os
import argparse
from utils import scoremap2bbox, parse_xml_to_dict, _convert_image_to_rgb, compute_AP, compute_F1, _transform_resize
from clip_text import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO, class_names_coco_stuff182_dict, coco_stuff_182_to_27
import warnings
warnings.filterwarnings("ignore")


def mask_attn(logits_coarse, logits, h, w, attn_weight):
    patch_size = 16
    candidate_cls_list = []
    logits_refined = logits.clone()
    
    logits_max = torch.max(logits, dim=0)[0]
        
    for tempid,tempv in enumerate(logits_max):
        if tempv > 0:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:,ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits
        mask = mask.reshape(h // patch_size, w // patch_size)
        
        box, cnt = scoremap2bbox(mask.detach().cpu().numpy(), threshold=temp_logits.mean(), multi_contour_eval=True)
        aff_mask = torch.zeros((mask.shape[0],mask.shape[1])).to(device)
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1,mask.shape[0] * mask.shape[1])
        trans_mat = attn_weight * aff_mask
        logits_refined_ccls = torch.matmul(trans_mat, logits_coarse[:,ccls:ccls+1])
        logits_refined[:, ccls] = logits_refined_ccls.squeeze()
    return logits_refined

def cwr(logits, logits_max, h, w, image, text_features):
    patch_size = 16
    input_size = 224
    stride = input_size // patch_size
    candidate_cls_list = []
    
    ma = logits.max()
    mi = logits.min()
    step = ma - mi
    if args.dataset == 'cocostuff':
        thres_abs = 0.1
    else:
        thres_abs = 0.5
    thres = mi + thres_abs*step
        
    for tempid,tempv in enumerate(logits_max):
        if tempv > thres:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:,ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits > 0.5
        mask = mask.reshape(h // patch_size, w // patch_size)
        
        horizontal_indicies = np.where(np.any(mask.cpu().numpy(), axis=0))[0]
        vertical_indicies = np.where(np.any(mask.cpu().numpy(), axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        
        y1 = max(y1, 0)
        x1 = max(x1, 0)
        y2 = min(y2, mask.shape[-2] - 1)
        x2 = min(x2, mask.shape[-1] - 1)
        if x1 == x2 or y1 == y2:
            return logits_max
        
        mask = mask[y1:y2, x1:x2]
        mask = mask.float()
        mask = mask[None, None, :, :]
        mask = F.interpolate(mask, size=(stride, stride), mode="nearest")
        mask = mask.squeeze()
        mask = mask.reshape(-1).bool()
        
        image_cut = image[:, :, int(y1*patch_size):int(y2*patch_size), int(x1*patch_size):int(x2*patch_size)]
        image_cut = F.interpolate(image_cut, size=(input_size, input_size), mode="bilinear", align_corners=False)
        cls_attn = 1 - torch.ones((stride*stride+1, stride*stride+1))
        for j in range(1, cls_attn.shape[1]):
            if not mask[j - 1]:
                cls_attn[0, j] = -1000

        image_features = model.encode_image_tagclip(image_cut, input_size, input_size, attn_mask=cls_attn)[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = model.logit_scale.exp()
        cur_logits = logit_scale * image_features @ text_features.t()
        cur_logits = cur_logits[:, 0, :]
        cur_logits = cur_logits.softmax(dim=-1).squeeze()
        cur_logits_norm = cur_logits[ccls]
        logits_max[ccls] = 0.5 * logits_max[ccls] + (1 - 0.5) * cur_logits_norm
            
    return logits_max


def classify():
    pred_label_id = []
    gt_label_id = []
    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, class_names, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    for im_idx, im in enumerate(tqdm(image_list)):
        image_path = os.path.join(args.img_root, im)
        
        label_id_list = all_label_list[im_idx]
        label_id_list = [int(lid) for lid in label_id_list]
        if args.dataset == 'cocostuff':
            label_id_list = [coco_stuff_182_to_171[int(lid)] for lid in label_id_list]
        gt_label_id.append(label_id_list)
        
        pil_img = Image.open(image_path)
        array_img = np.array(pil_img)
        ori_height, ori_width = array_img.shape[:2]
        if len(array_img.shape) == 2:
            array_img = np.stack([array_img, array_img, array_img], axis=2)
            pil_img = Image.fromarray(np.uint8(array_img))
        
        if model_type == 'clip':
            patch_size = 16
            preprocess = _transform_resize(int(np.ceil(int(ori_height) / patch_size) * patch_size), int(np.ceil(int(ori_width) / patch_size) * patch_size))
            image = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract image features
                h, w = image.shape[-2], image.shape[-1]
                
                image_features, attn_weight_list = model.encode_image_tagclip(image, h, w, attn_mask=1)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                

                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
                
                attn_vote = torch.stack(attn_weight, dim=0).squeeze()
                
                thres0 = attn_vote.reshape(attn_vote.shape[0], -1)
                thres0 = torch.mean(thres0, dim=-1).reshape(attn_vote.shape[0], 1, 1)
                thres0 = thres0.repeat(1, attn_vote.shape[1], attn_vote.shape[2])
                
                if args.dataset == 'cocostuff':
                    attn_weight = torch.stack(attn_weight, dim=0)[:-1]
                else:
                    attn_weight = torch.stack(attn_weight, dim=0)[8:-1]
                
                attn_cnt = attn_vote > thres0
                attn_cnt = attn_cnt.float()
                attn_cnt = torch.sum(attn_cnt, dim=0)
                attn_cnt = attn_cnt >= 4
                
                attn_weight = torch.mean(attn_weight, dim=0)[0]
                attn_weight = attn_weight * attn_cnt.float()
        
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()#torch.Size([1, 197, 81])
                logits = logits[:, 1:, :]
                logits = logits.softmax(dim=-1)
                logits_coarse = logits.squeeze()
                logits = torch.matmul(attn_weight, logits)
                logits = logits.squeeze()
                logits = mask_attn(logits_coarse, logits, h, w, attn_weight)

                logits_max = torch.max(logits, dim=0)[0]
                logits_max = logits_max[:NUM_CLASSES]
                logits_max = cwr(logits, logits_max, h, w, image, text_features)
                logits_max = logits_max.cpu().numpy()
            pred_label_id.append(logits_max)
    
        else:
            raise NotImplementedError()
    
    gt_one_hot = np.zeros((len(gt_label_id), NUM_CLASSES))
    for i in range(len(gt_label_id)):
        gt_ids = gt_label_id[i]
        for gt_id in gt_ids:
            gt_one_hot[i][gt_id] = 1
    
    predictions = torch.tensor(pred_label_id)
    labels = torch.tensor(gt_one_hot)
    
    # compute AP
    ap = compute_AP(predictions, labels)
    print('================================================')
    print('mAP: %.6f' % torch.mean(ap))
    
    # compute F1, P, R with specific relative threshold
    ma = predictions.max(dim=1)[0]
    mi = predictions.min(dim=1)[0]
    step = ma - mi
    if args.dataset == 'cocostuff':
        thres_abs = 0.1
    else:
        thres_abs = 0.5
    
    F1, P, R = compute_F1(predictions.clone(), labels.clone(), 'overall', thres_abs, use_relative=True)
    print('F1: %.6f, Precision: %.6f, Recall: %.6f' % (torch.mean(F1), torch.mean(P), torch.mean(R)))
    print('================================================\n')

    #save class labels
    if args.save_file:
        save_path = './output/{}_val_tagclip.txt'.format(args.dataset)
        print('>>>writing to {}'.format(save_path))
        thres_rel = mi + thres_abs * step
        with open(save_path, 'w') as f:
            for im_idx, im in enumerate(image_list):
                line = im.replace('.jpg','')
                for index, value in enumerate(pred_label_id[im_idx]):
                    if value > thres_rel[im_idx]:
                        line += " {}".format(index)
                if line == im.replace('.jpg',''):
                    line += " {}".format(np.argmax(pred_label_id[im_idx]))
                line += "\n"
                f.writelines(line)
   

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='voc2007', choices=['voc2007', 'voc2012', 'coco2014', 'coco2017', 'cocostuff'])
    parser.add_argument('--img_root', type=str, default='./datasets/VOC2007/JPEGImages')
    parser.add_argument('--split_file', type=str, default='./datasets/VOC2007/ImageSets/Main/test.txt')
    parser.add_argument('--model_path', type=str, default='ViT-B/16')
    parser.add_argument('--save_file', action="store_true")
    args = parser.parse_args()
    
    if args.dataset in ['voc2007', 'voc2012']:
        class_names = class_names_voc + BACKGROUND_CATEGORY_VOC
        NUM_CLASSES = len(class_names_voc)
    elif args.dataset in ['coco2014', 'coco2017']:
        class_names = class_names_coco + BACKGROUND_CATEGORY_COCO
        NUM_CLASSES = len(class_names_coco)
    else:
        coco_stuff_182_to_171 = {}
        cnt = 0
        for label_id in coco_stuff_182_to_27:
            if label_id + 1 in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:  # note that +1 is added
                continue
            coco_stuff_182_to_171[label_id] = cnt
            cnt += 1
            
        class_names = []
        for k in class_names_coco_stuff182_dict.keys():
            if k in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                continue
            class_names.append(class_names_coco_stuff182_dict[k])
        NUM_CLASSES = len(class_names)

    print('================================================')
    print('num_classes_dataset:', NUM_CLASSES)
    print('num_classes_all:', len(class_names))

    file_list = tuple(open(args.split_file, "r"))
    file_list = [id_.rstrip().split(" ") for id_ in file_list]
    image_list = [x[0] + '.jpg' for x in file_list]
    all_label_list = [x[1:] for x in file_list]
    
    print("num_images:", len(image_list))

    
    model_type = "clip"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print('================================================\n')
    model, _ = clip.load(args.model_path, device=device)
    model.eval()
    classify()