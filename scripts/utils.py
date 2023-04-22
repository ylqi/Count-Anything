import spacy
import numpy as np


nlp = spacy.load('en_core_web_sm')
def get_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    return noun_phrases


def dice_coef(source, target, smooth = 1.):
    source_f = source.flatten()
    target_f = target.flatten()
    
    intersection = np.sum(source_f * target_f)
    dice = (intersection + smooth) / (np.sum(source_f) + smooth)
    
    return dice, np.sum(source_f) < np.sum(target_f)