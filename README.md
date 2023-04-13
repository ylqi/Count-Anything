# Count Anything

### [Official repo](https://github.com/ylqi/Count-Anything)
> **[Count Anything](https://github.com/ylqi/Count-Anything)**  
> Liqi, Yan                            
> ZJU-CV, Zhejiang University / Fudan Univerisity

_**C**ount **A**nything (CA)_ project is a versatile image processing tool that combines the capabilities of [Segment Anything](https://segment-anything.com/), [Semantic-Segment-Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything), and [CLIP](https://arxiv.org/abs/2103.00020).
Our solution can count *any object* specified by users within an image.

### 🚄 Count Anything (CA) engine
![](./figures/CA_model.png)
The CA engine consists of three steps:
- **(I) Segement Anything.** Following [Semantic-Segment-Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything), CA engine crops a patch for each mask predicted by [Segment Anything](https://segment-anything.com/).
- **(II) Class Mixer.** To identify the masks that match the user’s text prompt, we add the text prompt as an additional class into the class list from the close-set datasets (COCO or ADE20K). 
- **(III) CLIP Encoders.** The CA engine uses CLIP image encoder and text encoder to assess if the text prompt is the best option among other classes. If yes, this mask is considered as an instance of the class given by the text prompt, and the count number is incremented by 1.

## Examples
![](./figures/example_1.png)

## 💻 Requirements
- Python 3.7+
- CUDA 11.1+

## 🛠️ Installation
```bash
conda env create -f environment.yaml
conda activate ca-env
```
## 🚀 Quick Start
### 1. Run [Segment Anything](https://segment-anything.com/) to get segmentation jsons for each image:
```bash
python scripts/amg.py --checkpoint sam_vit_h_4b8939.pth --input examples/AdobeStock_323574125.jpg --output output --model-type vit_h  --pred-iou-thresh 0.98 --crop-n-layers 0 --crop-nms-thresh 0.3 --box-nms-thresh 0.5 --stability-score-thresh 0.7 --convert-to-rle
```
### 2. Save the `.jpg` and `.json` in our `data/examples` folder:
```none
├── Count-Anything
|   ├── data
|   │   ├── examples
|   │   │   ├── AdobeStock_323574125.jpg
|   │   │   ├── AdobeStock_323574125.json
|   │   │   ├── ...
```

### 3. Run our Count Anything engine with 1 GPU:
```bash
python scripts/main.py --data_dir=data/examples --out_dir=output --world_size=1 --save_img --text_prompt shirt
```
The result is saved in `output` folder.

## 😄 Acknowledgement
- [Segment Anything](https://segment-anything.com/) provides the SA-1B dataset.
- [HuggingFace](https://huggingface.co/) provides code and pre-trained models.
- [Semantic-Segment-Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything) provides code.
- [CLIP](https://arxiv.org/abs/2103.00020) provide powerful semantic segmentation, image caption and classification models.

## 📜 Citation
If you find this work useful for your research, please cite our github repo:
```bibtex
@misc{chen2023semantic,
    title = {Count Anything},
    author = {Yan, Liqi},
    howpublished = {\url{https://github.com/ylqi/Count-Anything}},
    year = {2023}
}
```
