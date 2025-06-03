from google.colab import drive
import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image
import torch

# Initialize dataset
# Translate English captions of MS-COCO datasets into languages in lang_map using NLLB model
# And then, download the images of corresponding captions from MS-COCO 2017 train set.

# for colab users (ME)
drive.mount('/content/drive')
DRIVE_PATH = "/content/drive/MyDrive/CS371/Final Research/dataset/coco_nllb_dataset"
json_path = "/train2017/captions_train2017.json"

img_data_json_path = DRIVE_PATH + json_path  # coco annotation json
translated_captions_json_path = DRIVE_PATH + "/train2017/captions_train2017_translated_5000.json"
save_dir = DRIVE_PATH + "/train2017/images"

with open(DRIVE_PATH + json_path, 'r') as f:
    data_json = json.load(f)
    img_data = data_json['images']
    cap_data = data_json['annotations']

process_range = (0,15000) 
output_path = DRIVE_PATH + "/train2017/captions_train2017_translated_"+str(process_range[1])+".json"


lang_map = {
    'ko': 'kor_Hang',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'th': 'tha_Thai',
    'sw': 'swh_Latn',
    'bn': 'ben_Beng',
}

model_name = "facebook/nllb-200-distilled-600M"

"""#### translating captions using nllb model; only for initial setup"""

# for train set

def translate_caption(caption, target_lang_code):
    tokenizer.src_lang = "eng_Latn"
    encoded = tokenizer(caption, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(
        **encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

translated_captions = []

# 5000개만 처리
i=0
for ann in tqdm(cap_data[process_range[0]:process_range[1]]):
    if 'caption' in ann:
        original = ann['caption']
        #print("original english caption : ", original)
        translated_dict = {'id': ann['id'], 'caption_en': original, 'image_id' : ann['image_id']}
        for lang_code, nllb_code in lang_map.items():
            translated = translate_caption(original, nllb_code)
            translated_dict[f'caption_{lang_code}'] = translated
            #print("translated into : ",lang_code,translated)
        translated_captions.append(translated_dict)

    if i % 100 == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated_captions, f, ensure_ascii=False, indent=2)
    i+=1

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(translated_captions, f, ensure_ascii=False, indent=2)

print(f"Saved translated captions to {output_path}")

"""## 2. Get images using coco_url"""

import requests

def download_filtered_images(img_data_json_path, translated_captions_json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 이미지 메타데이터 로드
    with open(img_data_json_path, 'r', encoding='utf-8') as f:
        img_data = json.load(f)['images']

    # 번역된 캡션에서 id 리스트 로드
    with open(translated_captions_json_path, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    valid_ids = [item['id'] for item in captions]
    valid_img_ids = [cap['image_id'] for cap in cap_data[process_range[0]:process_range[1]] if cap['id'] in valid_ids]
    print(len(valid_ids), valid_img_ids)

    # 이미지 중에서 valid_ids에 포함된 것만 필터링
    filtered_imgs = [img for img in img_data if img['id'] in valid_img_ids]
    print(len(filtered_imgs))

    for img in tqdm(filtered_imgs):
        img_id = img['id']
        url = img['coco_url']
        file_path = os.path.join(save_dir, f"{img_id}.jpg")

        if os.path.exists(file_path):
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download image {img_id} from {url}: {e}")

# 사용 예시
img_data_json_path = DRIVE_PATH + json_path  # coco annotation json
translated_captions_json_path = output_path
save_dir = DRIVE_PATH + "/train2017/images"

download_filtered_images(img_data_json_path, translated_captions_json_path, save_dir)