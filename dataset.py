from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms


"""## 3. Dataset Class

"""

class MultilingualCocoDataset(Dataset):
    def __init__(self, captions_json_path, img_dir, lang_list, transform=None,num_data=5000):
        import json
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)[:num_data] # 리스트 형태로 [{...}, {...}, ...]
        self.img_dir = img_dir
        self.lang_list = lang_list

        # 기본 전처리 (정규화 제외)
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # 정규화는 별도로 저장 (학습 루프에서 필요할 때만 사용)
        self.normalize = transforms.Normalize(
            mean=image_preprocessor.image_mean,
            std=image_preprocessor.image_std
        )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = self.captions[idx]
        img_id = item['image_id'] # image_id 키 사용
        #print(self.captions[idx])

        # 이미지 경로: image_id.jpg
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # 기본 전처리만 적용 (Resize + ToTensor, 정규화는 제외)
        if self.base_transform:
            image = self.base_transform(image)

        if self.normalize:
            image = self.normalize(image)

        # 캡션 딕셔너리 생성
        captions = {}
        for lang in self.lang_list:
            key = f"caption_{lang}"
            captions[lang] = item.get(key, None)

        return image, captions