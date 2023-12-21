import numpy as np
import cv2
from lxml import etree
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torch
BICUBIC = InterpolationMode.BICUBIC
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def parse_xml_to_dict(xml):
    """
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}



def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform_resize(h, w):
    return Compose([
        #Resize(n_px, interpolation=BICUBIC),
        Resize((h,w), interpolation=BICUBIC),
        #CenterCrop(n_px),
        #RandomHorizontalFlip(1.0),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        #mask = label.abs() == 1
        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label, min=0, max=1)
        sorted_pred, sort_idx = prediction.sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp/num_pos
        prec = tp/(tp+fp)
        ap_cls = (tmp*prec).sum()/num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap



def compute_F1(predictions, labels, mode_F1, k_val, use_relative=False):
    if k_val >= 1:
        idx = predictions.topk(dim=1, k=k_val)[1]
        predictions.fill_(0)
        predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val, dtype=predictions.dtype).to(predictions.device))
    else:
        if use_relative:
            ma = predictions.max(dim=1)[0]
            mi = predictions.min(dim=1)[0]
            step = ma - mi
            thres = mi + k_val * step
        
            for i in range(predictions.shape[0]):
                predictions[i][predictions[i] > thres[i]] = 1
                predictions[i][predictions[i] <= thres[i]] = 0
        else:
            predictions[predictions > k_val] = 1
            predictions[predictions <= k_val] = 0
        
    if mode_F1 == 'overall':
        predictions = predictions.bool()
        labels = labels.bool()
        TPs = ( predictions &  labels).sum()
        FPs = ( predictions & ~labels).sum()
        FNs = (~predictions &  labels).sum()
        eps = 1.e-9
        Ps = TPs / (TPs + FPs + eps)
        Rs = TPs / (TPs + FNs + eps)
        p = Ps.mean()
        r = Rs.mean()
        f1 = 2*p*r/(p+r)
        
    
    elif mode_F1 == 'category':
        # calculate P and R
        predictions = predictions.bool()
        labels = labels.bool()
        TPs = ( predictions &  labels).sum(axis=0)
        FPs = ( predictions & ~labels).sum(axis=0)
        FNs = (~predictions &  labels).sum(axis=0)
        eps = 1.e-9
        Ps = TPs / (TPs + FPs + eps)
        Rs = TPs / (TPs + FNs + eps)
        p = Ps.mean()
        r = Rs.mean()
        f1 = 2*p*r/(p+r)
        
    elif mode_F1 == 'sample':
        # calculate P and R
        predictions = predictions.bool()
        labels = labels.bool()
        TPs = ( predictions &  labels).sum(axis=1)
        FPs = ( predictions & ~labels).sum(axis=1)
        FNs = (~predictions &  labels).sum(axis=1)
        eps = 1.e-9
        Ps = TPs / (TPs + FPs + eps)
        Rs = TPs / (TPs + FNs + eps)
        p = Ps.mean()
        r = Rs.mean()
        f1 = 2*p*r/(p+r)

    return f1, p, r