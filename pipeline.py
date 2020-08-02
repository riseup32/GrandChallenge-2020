#!/usr/bin/env python
# coding: utf-8

import os
import time
import warnings

import numpy as np
import torch
import librosa
import speech_recognition
import speech_model

from utils.tokenization import BertTokenizer
from utils.classifier_utils import KorNLIProcessor
from preprocessing import preprocessing

warnings.filterwarnings(action='ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_speech = speech_model.Classifier()
model_speech.load_state_dict(torch.load('./output/m_speech.pt'))
model_speech.to(device)
model_speech.eval()



tokenizer = BertTokenizer('./data/large_v2_32k_vocab.txt', max_len=128)

model_text = torch.load('./output/m_text.pt')
model_text = model_text.module
model_text.to(device)
model_text.eval()



processor = KorNLIProcessor()
output_mode = "classification"

label_list = processor.get_labels()
num_labels = len(label_list)



recognizer = speech_recognition.Recognizer()
recognizer.energy_threshold = 300



def speech_to_text(path):
    audio = speech_recognition.AudioFile(path)
    with audio as source:
        ex = recognizer.record(source)
        text = recognizer.recognize_google(audio_data=ex, language='ko-KR')
    return text



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids



def convert_example_to_feature(example, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    tokens_a = tokenizer.tokenize(example)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids)
    return feature



def predict_speech(x):
    return torch.mean((torch.sigmoid(model_speech(x)) >= 0.5).float()).item()




def predict_text(input_id, segment_id, input_mask):
    logit = model_text(input_id, segment_id, input_mask, labels=None)
    return torch.nn.Softmax(dim=-1)(logit)[0, 1].item()



def predict(x, input_id, segment_id, input_mask, threshold=0.5):
    pred_speech = predict_speech(x)
    pred_text = predict_text(input_id, segment_id, input_mask)
    pred = 0.5 * pred_speech + 0.5 * pred_text
    pred = 1 if pred >= threshold else 0
    return pred



for i in range(1, 10):
    start = time.time()
    
    try:
        spectrogram = preprocessing('./wav_data/sample_0%d.wav'%i, method='mfcc', sr=22050)
        spectrogram = torch.tensor(spectrogram, device=device).float()
        spectrogram = spectrogram.permute(0, 3, 1, 2)

        text = speech_to_text('./wav_data/sample_0%d.wav'%i)
        feature = convert_example_to_feature(text, label_list, 128, tokenizer, output_mode)

        input_id = torch.tensor([feature.input_ids], dtype=torch.long).to(device)
        input_mask = torch.tensor([feature.input_mask], dtype=torch.long).to(device)
        segment_id = torch.tensor([feature.segment_ids], dtype=torch.long).to(device)

        pred = predict(spectrogram, input_id, segment_id, input_mask)

        print('sample_%d:'%i, pred, '\t run time:', time.time()-start)
    except:
        pass




