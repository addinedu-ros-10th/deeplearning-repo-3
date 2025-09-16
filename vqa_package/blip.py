from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
from transformers import GenerationConfig

from PIL import Image
from datetime import datetime
import cv2
import torch
import json
import mysql.connector
import matplotlib.pyplot as plt

from deep_translator import GoogleTranslator

import os
import sys
import urllib.request
client_i = "x9uu8eegr6"
client_p = "jMDetzjJqUMABin25qi9qTLAMy94TxOMlqwkXjCO"
url = "https://papago.apigw.ntruss.com/nmt/v1/translation"
request = urllib.request.Request(url)
request.add_header("X-NCP-APIGW-API-KEY-ID",client_i)
request.add_header("X-NCP-APIGW-API-KEY",client_p)


from peft import PeftModel
from transformers import BlipForConditionalGeneration, BlipProcessor, GenerationConfig,  DisjunctiveConstraint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_ID = "Salesforce/blip-image-captioning-base"
ADAPTER_DIR = "./blip_lora_adapter"

proc = BlipProcessor.from_pretrained(ADAPTER_DIR, use_fast=True)  
base = BlipForConditionalGeneration.from_pretrained(BASE_ID)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device).float().eval()

qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor_c = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)


hazard_words = ["crosswalk", "car", "bollard", "bollards", 'pole', 'poles', 'bar', 'people', 'stairs', 'ribbon']
hazard_ids = [proc.tokenizer(w, add_special_tokens=False).input_ids for w in hazard_words]
constraints = [DisjunctiveConstraint(hazard_ids)]  # OR 제약

bad_phrases = ["in a crosswalk", "crossing", "driving", "parked", 'stopped', 'building', 'painted', 'it', 'light pole', 'lot']  # 의미 오류 유발 구문 차단
bad_ids = [proc.tokenizer(p, add_special_tokens=False).input_ids for p in bad_phrases]


def translate_enko(outputs):
    caption = proc.decode(outputs[0], skip_special_tokens=True)
    print("영어 :", caption)

    data = "source=en&target=ko&text=" + urllib.parse.quote(caption)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    response_body = response.read()
    parsed = json.loads(response_body)
    translated = parsed["message"]["result"]["translatedText"]
    print("한국어 :", translated) 

    return translated


def db_insert(input, translated):
    remote = mysql.connector.connect(
        host = 'database-1.ct0kcwawch43.ap-northeast-2.rds.amazonaws.com ',
        port = 3306,
        user = 'robot',
        password = '0310',
        database = 'bhc_database'
    )
    cur = remote.cursor(buffered=True)
    now = datetime.now()

    query = f"insert into vqa_log (question, answer) values ('{input}', '{translated}')"
    cur.execute(query)
    remote.commit()
    remote.close()

def caption_image(img_path:str):
    gen_cfg = GenerationConfig(
    num_beams=2,
    max_new_tokens=60,         # 최대 토큰
    do_sample=False,            # False는 그리디(안정적/단조로움). True는 높은 확률(더 사람같은 표현)
    repetition_penalty=1.3,   # 단어 반복 억제(같은 단어 확률 줄임)
    no_repeat_ngram_size=3,     # n-gram 반복 억제(연속된 n개 단어 다시 나오지않게 강제)
    bad_words_ids=bad_ids, constraints=constraints
    )
    raw_image = Image.open(img_path).convert("RGB")
    inputs = proc(images=raw_image, return_tensors="pt")
    
    model_dtype = next(model.parameters()).dtype  # torch.float32
    inputs = {k: (v.to(device, dtype=model_dtype) if v.dtype.is_floating_point else v.to(device))
            for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)
        result = translate_enko(outputs)
    # print(result)
    input = "상황 설명"
    return input, result


def question_image(img, question):
    gen_cfg = GenerationConfig(
        max_new_tokens=20,         # 최대 토큰
        do_sample=False,            # False는 그리디(안정적/단조로움). True는 높은 확률(더 사람같은 표현)
        num_beams = 3
    )
    hazard_words = ["crosswalk", 'traffic light', "car", "bollard", "bollards", 'pole', 'poles', 'bar', 'people', 'stairs', 'man', 'person']
    hazard_ids = [proc.tokenizer(w, add_special_tokens=False).input_ids for w in hazard_words]
    constraints = [DisjunctiveConstraint(hazard_ids)]  # OR 제약

    bad_phrases = ["in a crosswalk", "crossing", "driving", "parked", 'stopped', 'building', 'painted', 'it', 'light pole', 'parking meters']  # 의미 오류 유발 구문 차단
    bad_ids = [proc.tokenizer(p, add_special_tokens=False).input_ids for p in bad_phrases]

    input = '지금 그림에 ' + question
    print("한국어 :", input)
    translated = GoogleTranslator(source='ko', target='en').translate(input)
    print("영어 :", translated) 

    raw_image = Image.open(img).convert('RGB')
    inputs = processor_c(raw_image, translated, return_tensors="pt")

    out = qa_model.generate(**inputs,  length_penalty=1.0,generation_config=gen_cfg, bad_words_ids=bad_ids, constraints=constraints)
    caption = proc.decode(out[0], skip_special_tokens=True)
    print("영어 :", caption)

    translated = GoogleTranslator(source='en', target='ko').translate(caption)
    print("한국어 :", translated) 

    return input, translated