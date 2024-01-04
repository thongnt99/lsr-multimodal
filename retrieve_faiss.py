import argparse
import torch
from datasets import load_dataset
import time
import faiss
faiss.omp_set_num_threads(1)
parser = argparse.ArgumentParser(description="LSR Index Pisa")
parser.add_argument("--data", type=str,
                    default="lsr42/mscoco-blip-dense")
args = parser.parse_args()
dataset = load_dataset(args.data, data_files={"img_emb": "img_embs.parquet",
                                              "text_emb": "text_embs.parquet"}, keep_in_memory=True).with_format("numpy")

index = faiss.IndexHNSWFlat(256, 32, 0)
index.train(dataset["img_emb"]["emb"])
index.add(dataset["img_emb"]["emb"])
queries = dataset["text_emb"]["emb"][:2000]
start = time.time()
D, I = index.search(queries, 1000)
end = time.time()
total_time = end - start
print(f"Total running time: {total_time} seconds")
print(f"s/q: {total_time*1.0/len(queries)}")
print(f"q/s: {len(queries)*1.0/total_time}")
