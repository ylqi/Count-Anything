import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

def clip_classification(image, class_list, top_k, clip_processor, clip_model, rank):
    inputs = clip_processor(text=class_list, images=image, return_tensors="pt", padding=True).to(rank)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # if top_k == 1:
    #     class_name = class_list[probs.argmax().item()]
    #     prob = probs[0][probs.argmax().item()]
    #     return class_name, prob
    # else:
    top_k_indices = probs.topk(top_k, dim=1).indices[0]
    top_k_class_names = [class_list[index] for index in top_k_indices]
    # print("probs: ", probs)
    top_k_probs = [probs[0][index] for index in top_k_indices]
    return top_k_class_names, top_k_probs
    

def clip_text_features(text, clip_model, rank):
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    inputs = tokenizer([text], padding=True, return_tensors="pt").to(rank)
    text_features = clip_model.get_text_features(**inputs)
    return text_features[0]