from torch.utils.data import Dataset
import torch


class TrainDataset(Dataset):
    def __init__(self, texts_dict,  txtid2row, imgid2row, text_embs, img_embs, text_image_pairs):
        super().__init__()
        self.texts_dict = texts_dict
        self.txtid2row = txtid2row
        self.imgid2row = imgid2row
        self.text_embs = text_embs
        self.img_embs = img_embs
        self.text_image_pairs = text_image_pairs

    def __len__(self):
        return len(self.text_image_pairs)

    def __getitem__(self, index):
        text_id, image_id = self.text_image_pairs[index]
        text = self.texts_dict[text_id]
        text_emb = self.text_embs[self.txtid2row[text_id]]
        img_emb = self.img_embs[self.imgid2row[image_id]]
        return {"type": "train", "text": text, "text_emb": torch.tensor(text_emb), "img_emb": torch.tensor(img_emb)}


class VectorCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        if batch[0]["type"] == "train":
            batch_texts = []
            batch_txt_embs = []
            batch_img_embs = []
            for item in batch:
                batch_texts.append(item["text"])
                batch_txt_embs.append(item["text_emb"])
                batch_img_embs.append(item["img_emb"])
            batch_txt_embs = torch.stack(batch_txt_embs, dim=0)
            batch_img_embs = torch.stack(batch_img_embs, dim=0)
            batch_texts = self.tokenizer(
                batch_texts, truncation=True, padding=True, return_tensors="pt")
            return batch_texts, batch_txt_embs, batch_img_embs
        elif batch[0]["type"] == "text":
            batch_text_ids = []
            batch_texts = []
            batch_txt_embs = []
            for item in batch:
                batch_text_ids.append(item["text_id"])
                batch_texts.append(item["text"])
                batch_txt_embs.append(item["text_emb"])
            batch_txt_embs = torch.stack(batch_txt_embs, dim=0)
            batch_texts = self.tokenizer(
                batch_texts, truncation=True, padding=True, return_tensors="pt")
            return batch_text_ids, batch_texts, batch_txt_embs
        elif batch[0]["type"] == "image":
            batch_img_ids = []
            batch_img_embs = []
            for item in batch:
                batch_img_ids.append(item["img_id"])
                batch_img_embs.append(item["img_emb"])
            batch_img_embs = torch.stack(batch_img_embs, dim=0)
            return batch_img_ids, batch_img_embs


class TextCollection(Dataset):
    def __init__(self, ids, texts, txtid2row, txt_embs):
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.txtid2row = txtid2row
        self.txt_embs = txt_embs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        item_text = self.texts[index]
        item_emb = self.txt_embs[self.txtid2row[item_id]]
        return {"type": "text", "text_id": item_id, "text": item_text, "text_emb": torch.tensor(item_emb)}


class ImageCollection(Dataset):
    def __init__(self, ids, imgid2row, img_embs):
        super().__init__()
        self.ids = ids
        self.imgid2row = imgid2row
        self.img_embs = img_embs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        item_emb = self.img_embs[self.imgid2row[item_id]]
        return {"type": "image", "img_id": item_id, "img_emb": torch.tensor(item_emb)}
