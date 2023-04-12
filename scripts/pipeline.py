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

def semantic_annotation_pipeline(filename, data_path, output_path, rank, save_img=False, scale_small=1.2, scale_large=1.6, scale_huge=3.0,
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

    all_patch_small = []
    all_patch_huge = []
    all_local_class_list = []
    all_valid_mask_huge_crop = []
    print("Pipline: Predicting with open-vocabuary classes...")
    for ann in anns:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        patch_small = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                scale=scale_small)
        patch_large = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                scale=scale_large)
        patch_huge = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                scale=scale_large)
        valid_mask_huge_crop = mmcv.imcrop(valid_mask.numpy(), np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                    scale=scale_large)
        all_patch_small.append(patch_small)
        all_patch_huge.append(patch_huge)
        all_valid_mask_huge_crop.append(valid_mask_huge_crop)
    
    if text_prompt is not None:
        ade20k_classes = json.load(open("datasets/shi-labs/oneformer_demo/ade20k_panoptic.json", 'r'))
        local_class_names = set([ade20k_classes[key]["name"] for key in ade20k_classes.keys()])
        local_class_names = set.union(local_class_names, set([text_prompt]))
        # print("local_class_list: ", local_class_names)
        for ann in anns:
            all_local_class_list.append(list(local_class_names))
    else:
        print("Pipline: Predicting coco classes...")
        # oneformer_coco_model.to(rank)
        # class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(img),oneformer_coco_processor,oneformer_coco_model, rank)
        oneformer_ade20k_model.to(rank)
        class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(img),oneformer_ade20k_processor,oneformer_ade20k_model, rank)

        blip_model.to(rank)
        print("Pipline: Predicting open-vocabuary classes...")
        for ann in anns:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            # get the class ids of the valid pixels
            # coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
            ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
            # top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
            top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
            local_class_names = set()
            local_class_names = set.union(local_class_names, set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in top_k_ade20k_propose_classes_ids]))
            # local_class_names = set.union(local_class_names, set(([CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in top_k_coco_propose_classes_ids])))
            
            op_class_list = open_vocabulary_classification_blip(patch_large,blip_processor, blip_model, rank)
            local_class_list = list(set.union(local_class_names, set(op_class_list))) # , set(refined_imagenet_class_names)
            all_local_class_list.append(local_class_list)
    
    clip_model.to(rank)
    all_mask_categories = []
    print("Pipline: Predicting clip scores...")
    for patch_small, local_class_list in zip(all_patch_small, all_local_class_list):
        mask_categories = clip_classification(patch_small, local_class_list, 3 if len(local_class_list)> 3 else len(local_class_list), clip_processor, clip_model, rank)
        all_mask_categories.append(mask_categories)
    
    clipseg_model.to(rank)

    if text_prompt is not None:
        print("Pipline: Counting...")
        for ann, mask_categories in zip(anns, all_mask_categories):
            # print("mask_categories: ", mask_categories)
            if text_prompt == mask_categories[0]:
                ann['class_name'] = "%s(id: %d)" % (text_prompt, count + 1)
                ann['class_proposals'] = mask_categories
                class_names.append(ann['class_name'])
                bitmasks.append(maskUtils.decode(ann['segmentation']))
                count += 1
    else:
        print("Pipline: Predicting clipseg top-1...")
        for ann, patch_huge, mask_categories, valid_mask_huge_crop in zip(anns, all_patch_huge, all_mask_categories, all_valid_mask_huge_crop):
            class_ids_patch_huge = clipseg_segmentation(patch_huge, mask_categories, clipseg_processor, clipseg_model, rank).argmax(0)
            top_1_patch_huge = torch.bincount(class_ids_patch_huge[torch.tensor(valid_mask_huge_crop)].flatten()).topk(1).indices
            top_1_mask_category = mask_categories[top_1_patch_huge.item()]

            ann['class_name'] = str(top_1_mask_category)
            ann['class_proposals'] = mask_categories
            class_names.append(ann['class_name'])
            bitmasks.append(maskUtils.decode(ann['segmentation']))
        
    mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
    if save_img:
        imshow_det_bboxes(img,
                    bboxes=None,
                    labels=np.arange(len(bitmasks)),
                    segms=np.stack(bitmasks),
                    class_names=class_names,
                    font_size=25,
                    show=False,
                    out_file=os.path.join(output_path, filename+'_class_name.png'))
        print(os.path.join(output_path, filename+'_class_name.png'))
    
    if text_prompt is not None:
        print("Detect %s count: %d" % (text_prompt, count))