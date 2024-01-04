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

parser = argparse.ArgumentParser(description="LSR Index Pisa")
parser.add_argument("--data", type=str,
                    default="lsr42/mscoco-blip-dense")
parser.add_argument("--batch_size", type=int,
                    default=1024, help="eval batch size")
parser.add_argument(
    "--model", type=str, default="lsr42/d2s_mscoco-blip-dense_q_reg_0.001_d_reg_0.001")
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

sparse_images = []
image_ids = []
image_outputs = []
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
        image_outputs.append(batch_sparse.to("cpu"))
image_outputs = torch.cat(image_outputs, 0)
with Pool(18) as p:
    sparse_images = p.starmap(create_json_doc, list(
        zip(image_ids, image_topk_toks, image_topk_weights)))
print(sparse_images[0])
index_name = f"./indexes/{args.data.replace('/','_')}/{args.model.replace('/','_')}"
index = PisaIndex(index_name, stemmer='none', threads=8)
indexer = index.toks_indexer(mode="overwrite")
indexer.index(sparse_images)
sparse_texts = []

meta_data = json.load(open(hf_hub_download(
    repo_id=args.data, repo_type="dataset", filename="dataset_meta.json")))

id2text = {}
for image in tqdm(meta_data['images'], desc="Processing meta data."):
    captions = [sent["raw"] for sent in image["sentences"]]
    caption_ids = [str(sent["sentid"]) for sent in image["sentences"]]
    id2text.update(dict(zip(caption_ids, captions)))

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
# qid2rep = {st["qid"]: st["query_toks"] for st in sparse_texts}
# did2rep = {si["docno"]: si["toks"] for si in sparse_images}
qid2index = dict(zip(text_ids, range(len(text_ids))))
did2index = dict(zip(image_ids, range(len(image_ids))))
lsr_searcher = index.quantized()
start = time.time()
res = lsr_searcher(sparse_texts_no_expansion)
for idx, pair in res.iterrows():
    if pair["rank"] >= 100:
        continue
    else:
        query_id = pair["qid"]
        doc_id = pair["docno"]
        qi = qid2index[query_id]
        di = did2index[doc_id]
        score = (text_outputs[qi] * image_outputs[di]).sum()

end = time.time()
total_time = end - start
print(f"Total running time: {total_time} seconds")
print(f"s/q: {total_time*1.0/len(sparse_texts)}")
print(f"q/s: {len(sparse_texts)*1.0/total_time}")
