import os
import argparse

from pipeline import semantic_annotation_pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.distributed as dist
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--out_dir', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--text_prompt', type=str, default=None, help='input object name to count')
    parser.add_argument('--area_min_thres', type=str, default=0.0015, help='input object size threshold to count')
    args = parser.parse_args()
    return args
    
def main(rank, args):
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    if args.text_prompt is None:
        oneformer_ade20k_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        oneformer_coco_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        clipseg_processor.image_processor.do_resize = False
    else:
        oneformer_ade20k_processor = None
        oneformer_ade20k_model = None

        oneformer_coco_processor = None
        oneformer_coco_model = None

        blip_processor = None
        blip_model = None

        clipseg_processor = None
        clipseg_model = None
    
    filenames = [fn_[:-5] for fn_ in os.listdir(args.data_dir) if '.json' in fn_]
    print("Load filenames: ", filenames)
    local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    for file_name in local_filenames:
        semantic_annotation_pipeline(file_name, args.data_dir, args.out_dir, rank, save_img=args.save_img, area_min_thres=args.area_min_thres,
                                    clip_processor=clip_processor, clip_model=clip_model,
                                    oneformer_ade20k_processor=oneformer_ade20k_processor, oneformer_ade20k_model=oneformer_ade20k_model,
                                    oneformer_coco_processor=oneformer_coco_processor, oneformer_coco_model=oneformer_coco_model,
                                    blip_processor=blip_processor, blip_model=blip_model, 
                                    clipseg_processor=clipseg_processor, clipseg_model=clipseg_model, text_prompt=args.text_prompt)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)