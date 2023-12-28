from huggingface_hub import HfApi
import pandas as pd
import json
import torch
mscoco_img_embs = torch.load("/projects/0/guse0488/dataset/mscoco/img_embs.pt")
mscoco_img_ids = json.load(
    open("/projects/0/guse0488/dataset/mscoco/img_ids.json"))
mscoco_img_df = pd.DataFrame({"id": mscoco_img_ids, "emb": mscoco_img_embs})
mscoco_img_df.to_parquet(
    "/projects/0/guse0488/dataset/mscoco/blip_img_embs.parquet")
api = HfApi()
api.upload_file(
    path_or_fileobj="/projects/0/guse0488/dataset/mscoco/blip_img_embs.parquet",
    path_in_repo="img_embs.parquet",
    repo_id="https://huggingface.co/datasets/lsr42/mscoco-blip-dense",
    repo_type="dataset",
)
