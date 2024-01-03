import argparse
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from model import D2SModel
from tqdm import tqdm
from transformers import AutoTokenizer
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import time

parser = argparse.ArgumentParser(description="LSR Index Pisa")
parser.add_argument("--data", type=str,
                    default="lsr42/mscoco-blip-dense")
parser.add_argument("--batch_size", type=int,
                    default=1024, help="eval batch size")
parser.add_argument(
    "--model", type=str, default="lsr42/d2s_mscoco-blip-dense_q_reg_0.001_d_reg_0.001")
args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = load_dataset(args.data, data_files={"img_emb": "img_embs.parquet",
                                              "text_emb": "text_embs.parquet"}, keep_in_memory=True).with_format("torch")

img_dataloader = DataLoader(dataset["img_emb"], batch_size=args.batch_size)
text_dataloader = DataLoader(dataset["text_emb"], batch_size=args.batch_size)

model = D2SModel.from_pretrained(args.model).to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sparse_images = []
for batch in tqdm(img_dataloader, desc="Encode images"):
    batch_ids = batch["id"]
    batch_dense = batch["emb"].to(device)
    with torch.no_grad():
        batch_sparse = model(batch_dense)
        max_k = (batch_sparse > 0).sum(dim=1).max().item()
        batch_topk_weights, batch_topk_indices = batch_sparse.topk(
            max_k, dim=1)
    for (img_id, topk_indices, topk_weights) in zip(batch_ids, batch_topk_indices, batch_topk_weights):
        topk_weights = (topk_weights*100).to("cpu").tolist()
        topk_toks = tokenizer.convert_ids_to_tokens(topk_indices)
        sparse_images.append(
            {"docno": img_id, "toks": {tok: w for tok, w in zip(topk_toks, topk_weights) if w > 0}})
print(sparse_images[0])
sparse_texts = []
for batch in tqdm(text_dataloader, desc="Encode texts"):
    batch_ids = batch["id"]
    batch_dense = batch["emb"].to(device)
    with torch.no_grad():
        batch_sparse = model(batch_dense)
        max_k = (batch_sparse > 0).sum(dim=1).max().item()
        batch_topk_indices, batch_topk_weights = batch_sparse.topk(
            max_k, dim=1)
    for (img_id, topk_indices, topk_weights) in zip(batch_ids, batch_topk_indices, batch_topk_weights):
        topk_weights = topk_weights.to("cpu").tolist()
        topk_toks = tokenizer.convert_ids_to_tokens(topk_indices)
        sparse_texts.append(
            {"qid": img_id, "query_toks": dict(zip(topk_toks, topk_weights))})
index_name = f"./indexes/{args.data.replace('/','_')}/{args.model.replace('/','_')}"
index = PisaIndex(index_name, stemmer='none')
indexer = index.toks_indexer(mode="overwrite")
indexer.index(sparse_images)
lsr_searcher = index.quantized()
start = time.time()
res = lsr_searcher(sparse_texts)
end = time.time()
total_time = end - start
print(f"Total running time: {total_time} seconds")
print(f"s/q: {total_time*1.0/len(sparse_texts)}")
print(f"q/s: {len(sparse_texts)*1.0/total_time}")
