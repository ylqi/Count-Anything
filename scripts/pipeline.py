import os
import torch
from PIL import Image
import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import pycocotools.mask as maskUtils
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from clip import clip_classification, clip_text_features
from clipseg import clipseg_segmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation
from blip import open_vocabulary_classification_blip

import torch.nn.functional as F
import json

from tqdm import tqdm

from utils import dice_coef

def semantic_annotation_pipeline(filename, data_path, output_path, rank, save_img=False, scale_small=1.2, scale_large=1.6, scale_huge=3.0, area_min_thres=0.0015,
                                 clip_processor=None,
                                 clip_model=None,
                                 oneformer_ade20k_processor=None,
                                 oneformer_ade20k_model=None,
                                 oneformer_coco_processor=None,
                                 oneformer_coco_model=None,
                                 blip_processor=None,
                                 blip_model=None,
                                 clipseg_processor=None,
                                 clipseg_model=None,
                                 text_prompt=None):
    count = 0
    anns = mmcv.load(os.path.join(data_path, filename+'.json'))
    img = mmcv.imread(os.path.join(data_path, filename+'.jpg'))
    bitmasks, class_names = [], []

    all_valid_mask = []
    all_patch_small = []
    all_patch_huge = []
    all_local_class_list = []
    all_valid_mask_huge_crop = []
    print("Pipline: Predicting with open-vocabuary classes...")
    for ann in anns:
        valid_mask = maskUtils.decode(ann['segmentation'])
        if np.sum(valid_mask.flatten()) < area_min_thres * valid_mask.shape[0] * valid_mask.shape[1]:
            continue
        patch_small = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                scale=scale_small)
        patch_large = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                scale=scale_large)
        patch_huge = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                scale=scale_large)
        valid_mask_huge_crop = mmcv.imcrop(valid_mask, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                    scale=scale_large)
        all_valid_mask.append(valid_mask)
        all_patch_small.append(patch_small)
        all_patch_huge.append(patch_huge)
        all_valid_mask_huge_crop.append(valid_mask_huge_crop)
    
    if text_prompt is not None:
        ade20k_classes = json.load(open("datasets/shi-labs/oneformer_demo/ade20k_panoptic.json", 'r'))
        local_class_names = set([ade20k_classes[key]["name"] for key in ade20k_classes.keys()])
        # imagenet_classes = json.load(open("datasets/imagenet-1k/dataset_infos.json", 'r'))["default"]["features"]["label"]["names"]
        # local_class_names = set([class_text.split(', ')[0] for class_text in imagenet_classes])

        if text_prompt in local_class_names:
            top_k = int(0.1 * len(local_class_names))
        else:
            top_k = 1
        
        local_class_names = set.union(local_class_names, set([text_prompt]))
        # print("local_class_list: ", local_class_names)
        for valid_mask in all_valid_mask:
            all_local_class_list.append(list(local_class_names))
    else:
        print("Pipline: Predicting coco classes...")
        # oneformer_coco_model.to(rank)
        class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(img),oneformer_coco_processor,oneformer_coco_model, rank)
        oneformer_ade20k_model.to(rank)
        class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(img),oneformer_ade20k_processor,oneformer_ade20k_model, rank)

        top_k = 3

        blip_model.to(rank)
        print("Pipline: Predicting open-vocabuary classes...")
        for valid_mask in all_valid_mask:
            # get the class ids of the valid pixels
            coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
            ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
            top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
            top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
            local_class_names = set()
            local_class_names = set.union(local_class_names, set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in top_k_ade20k_propose_classes_ids]))
            local_class_names = set.union(local_class_names, set(([CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in top_k_coco_propose_classes_ids])))
            
            op_class_list = open_vocabulary_classification_blip(patch_large,blip_processor, blip_model, rank)
            local_class_list = list(set.union(local_class_names, set(op_class_list))) # , set(refined_imagenet_class_names)
            all_local_class_list.append(local_class_list)
    
    clip_model.to(rank)
    all_mask_categories = []
    all_probs = []
    print("Pipline: Predicting constrastive loss with clip...")
    for patch_small, local_class_list in tqdm(zip(all_patch_small, all_local_class_list), total=len(all_local_class_list)):
        mask_categories, probs = clip_classification(patch_small, local_class_list, top_k if len(local_class_list)> top_k else len(local_class_list), clip_processor, clip_model, rank)
        all_mask_categories.append(mask_categories)
        all_probs.append([prob.cpu().detach().item() for prob in probs])
        del probs
    
    if text_prompt is not None:
        print("Pipline: Counting...")
        for valid_mask, mask_categories, probs in zip(all_valid_mask, all_mask_categories, all_probs):
            # print("mask_categories: ", mask_categories)
            if text_prompt in mask_categories and probs[mask_categories.index(text_prompt)] > 0.01:
                # ann['class_name'] = text_prompt
                # ann['class_proposals'] = mask_categories
                class_names.append(text_prompt)
                bitmasks.append(valid_mask)
                count += 1
    else:
        clipseg_model.to(rank)
        print("Pipline: Predicting clipseg top-1...")
        for valid_mask, patch_huge, mask_categories, valid_mask_huge_crop in zip(all_valid_mask, all_patch_huge, all_mask_categories, all_valid_mask_huge_crop):
            class_ids_patch_huge = clipseg_segmentation(patch_huge, mask_categories, clipseg_processor, clipseg_model, rank).argmax(0)
            top_1_patch_huge = torch.bincount(class_ids_patch_huge[torch.tensor(valid_mask_huge_crop)].flatten()).topk(1).indices
            top_1_mask_category = mask_categories[top_1_patch_huge.item()]

            # ann['class_name'] = str(top_1_mask_category)
            # ann['class_proposals'] = mask_categories
            class_names.append(top_1_mask_category)
            bitmasks.append(valid_mask)
    
    print("Pipline: Remove duplicates...")
    del_list = []
    for idx in tqdm(range(len(bitmasks))):
        bitmask = bitmasks[idx]
        for i, ref_bitmask in enumerate(bitmasks):
            if i == idx:
                continue
            dice, larger = dice_coef(bitmask, ref_bitmask)
            if dice > 0.4 and larger:
                del_list.append(idx)
                break
    del_list.reverse()
    for idx in del_list:
        bitmasks.pop(idx)
        class_names.pop(idx)

    # mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
    if save_img:
        imshow_det_bboxes(img,
                    bboxes=None,
                    labels=np.arange(len(bitmasks)),
                    segms=np.stack(bitmasks),
                    class_names=["%s (id: %d)" % (class_name, i + 1) for i, class_name in enumerate(class_names)],
                    font_size=25,
                    show=False,
                    out_file=os.path.join(output_path, filename+'_class_name.png'))
        print("Saved to: ", os.path.join(output_path, filename+'_class_name.png'))
    
    if text_prompt is not None:
        print("Detect %s count: %d" % (text_prompt, len(bitmasks)))