from pathlib import Path
from multiprocessing import Pool
import argparse
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from model import D2SModel
from tqdm import tqdm
from transformers import AutoTokenizer
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import time
import json
import numpy as np

parser = argparse.ArgumentParser(description="LSR Index Pisa")
parser.add_argument("--data", type=str,
                    default="lsr42/mscoco-blip-dense")
parser.add_argument("--batch_size", type=int,
                    default=1024, help="eval batch size")
parser.add_argument(
    "--model", type=str, default="lsr42/d2s_mscoco-blip-dense_q_reg_0.001_d_reg_0.001")
parser.add_argument(
    "--topk", type=int, default=10)
parser.add_argument(
    "--mode", type=str, default="no_exp", help="Retrieval mode: exp, no_exp, hybrid")
args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_json_doc(doc_id, topk_toks, topk_weights):
    doc = {"docno": doc_id, "toks": {tok: w for tok,
                                     w in zip(topk_toks, topk_weights) if w > 0}}
    return doc


def create_json_query(query_id, topk_toks, topk_weights):
    query = {"qid": query_id, "query_toks": {tok: w for tok,
                                             w in zip(topk_toks, topk_weights) if w > 0}}
    return query


dataset = load_dataset(args.data, data_files={"img_emb": "img_embs.parquet",
                                              "text_emb": "text_embs.parquet"}, keep_in_memory=True).with_format("torch")
img_dataloader = DataLoader(dataset["img_emb"], batch_size=args.batch_size)
text_dataloader = DataLoader(dataset["text_emb"], batch_size=args.batch_size)
model = D2SModel.from_pretrained(args.model).to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
index_dir = Path(
    f"./indexes/{args.data.replace('/','_')}/{args.model.replace('/','_')}")
index = PisaIndex(str(index_dir), stemmer='none', threads=1)
sparse_image_path = index_dir/"sparse_images.json"
if sparse_image_path.exists():
    sparse_images = json.load(open(sparse_image_path))
else:
    sparse_images = []
    image_ids = []
    image_topk_toks = []
    image_topk_weights = []
    for batch in tqdm(img_dataloader, desc="Encode images"):
        batch_ids = batch["id"]
        batch_dense = batch["emb"].to(device)
        with torch.no_grad():
            batch_sparse = model(batch_dense)
            max_k = (batch_sparse > 0).sum(dim=1).max().item()
            batch_topk_weights, batch_topk_indices = batch_sparse.topk(
                max_k, dim=1)
            batch_topk_toks = [tokenizer.convert_ids_to_tokens(
                list_tok_ids) for list_tok_ids in batch_topk_indices.to("cpu")]
            image_ids.extend(batch_ids)
            image_topk_toks.extend(batch_topk_toks)
            image_topk_weights.extend(batch_topk_weights.to("cpu").tolist())
    with Pool(18) as p:
        sparse_images = p.starmap(create_json_doc, list(
            zip(image_ids, image_topk_toks, image_topk_weights)))
    indexer = index.toks_indexer(mode="overwrite")
    indexer.index(sparse_images)
    json.dump(sparse_images, open(sparse_image_path, "w"))
print(sparse_images[0])
sparse_texts = []
sparse_texts_path = index_dir/"sparse_texts.json"
if sparse_texts_path.exists():
    sparse_texts = json.load(open(sparse_texts_path))
else:
    text_ids = []
    text_topk_toks = []
    text_outputs = []
    text_topk_weights = []
    for batch in tqdm(text_dataloader, desc="Encode texts"):
        batch_ids = batch["id"]
        batch_dense = batch["emb"].to(device)
        with torch.no_grad():
            batch_sparse = model(batch_dense)
            max_k = (batch_sparse > 0).sum(dim=1).max().item()
            batch_topk_weights, batch_topk_indices = batch_sparse.topk(
                max_k, dim=1)
        batch_topk_toks = [tokenizer.convert_ids_to_tokens(
            list_tok_ids) for list_tok_ids in batch_topk_indices.to("cpu")]
        text_ids.extend(batch_ids)
        text_topk_toks.extend(batch_topk_toks)
        text_topk_weights.extend(batch_topk_weights.to("cpu").tolist())
        text_outputs.append(batch_sparse.to("cpu"))
        break
    with Pool(18) as p:
        sparse_texts = p.starmap(create_json_query, list(
            zip(text_ids, text_topk_toks, text_topk_weights)))
    json.dump(sparse_texts, open(sparse_texts_path, "w"))


class ForwardScorer:
    def __init__(self, tokenizer, sparse_texts, sparse_images):
        self.text_2_row = {}
        self.img_2_row = {}
        self.text_tok_indices = []
        self.text_tok_weights = []
        self.img_tok_indices = []
        self.img_tok_weights = []
        max_tok = 0
        for idx, stext in tqdm(sparse_texts, desc="building forward index"):
            self.text_2_row[stext["qid"]] = idx
            tok2w = stext["query_toks"]
            toks = list(tok2w.keys())
            tok_weights = list(tok2w.values())
            tok_ids = tokenizer.convert_tokens_to_ids(toks)
            max_tok = max(max_tok, len(tok_ids))
            self.text_tok_indices.append(np.array(tok_ids))
            self.text_tok_weights.append(np.array(tok_weights))
        max_tok = 0
        for idx, simg in tqdm(sparse_images, desc="building forward index"):
            self.img_2_row[simg["docno"]] = idx
            tok2w = simg["toks"]
            toks = list(tok2w.keys())
            tok_weights = list(tok2w.values())
            tok_ids = tokenizer.convert_tokens_to_ids(toks)
            max_tok = max(max_tok, len(tok_ids))
            self.img_tok_indices.append(np.array(tok_ids))
            self.img_tok_weights.append(np.array(tok_weights))

    def score(self, q_id, d_id):
        q_row = self.text_2_row[q_id]
        qtok_ids = np.expand_dims(self.text_tok_indices[q_row], axis=1)
        qtok_weights = self.text_tok_weights[q_row]
        drow = self.img_2_row[d_id]
        dtok_ids = np.expand_dims(self.img_tok_indices[drow], axis=0)
        dtok_weights = self.img_tok_weights[drow]
        mask = (qtok_ids == dtok_ids)
        x, y = np.nonzero(mask)
        score = np.dot(qtok_weights[x], dtok_weights[y])
        return score


if args.mode == "exp":
    lsr_searcher = index.quantized(num_results=args.topk)
    start = time.time()
    res = lsr_searcher(sparse_texts)
    end = time.time()
    total_time = end - start
else:
    meta_data = json.load(open(hf_hub_download(
        repo_id=args.data, repo_type="dataset", filename="dataset_meta.json")))
    id2text = {}
    for image in tqdm(meta_data['images'], desc="Processing meta data."):
        captions = [sent["raw"] for sent in image["sentences"]]
        caption_ids = [str(sent["sentid"]) for sent in image["sentences"]]
        id2text.update(dict(zip(caption_ids, captions)))
    sparse_texts_no_expansion = []
    for st in sparse_texts:
        qid = st["qid"]
        qtext = id2text[qid]
        tokens = tokenizer.tokenize(qtext)
        toks = {tok: st["query_toks"][tok]
                for tok in tokens if tok in st["query_toks"]}
        sparse_texts_no_expansion.append({"qid": qid, "query_toks": toks})
    print(sparse_texts[0])
    print(sparse_texts_no_expansion[0])

    lsr_searcher = index.quantized(num_results=args.topk)
    if args.mode == "hybrid":
        forward_scorer = ForwardScorer(tokenizer, sparse_texts, sparse_images)
        # image_forward = {}
        # text_forward = {}
        # for image in tqdm(sparse_images, desc="Buiding forward indexing for image collection"):
        #     image_forward[image["docno"]] = image["toks"]
        # for text in sparse_texts:
        #     text_forward[text["qid"]] = text["query_toks"]
    start = time.time()
    res = lsr_searcher(sparse_texts_no_expansion)
    if args.mode == "hybrid":
        for idx, row in res.iterrows():
            row["score"] = forward_scorer.score(row["qid"], row["docno"])
    end = time.time()
    total_time = end - start
print(f"Total running time: {total_time} seconds")
print(f"s/q: {total_time*1.0/len(sparse_texts)}")
print(f"q/s: {len(sparse_texts)*1.0/total_time}")
