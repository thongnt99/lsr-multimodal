from loss import BICELoss
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from collections import defaultdict
import torch
from torch import nn
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import ir_measures
from ir_measures import *
from collections import OrderedDict
import torch.nn.functional as F
from regularizer import *
from dataset import *
from model import D2SModel
from pathlib import Path
from utils import write_trec_file
from utils import cal_correaltion
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Training Text-Image LSR models")
parser.add_argument("--data", type=str,
                    default="lsr42/mscoco-blip-dense")
parser.add_argument("--train_batch_size", type=int,
                    default=512, help="train batch size")
parser.add_argument("--eval_batch_size", type=int,
                    default=1024, help="eval batch size")
parser.add_argument("--temp", type=float,
                    default=1e-3, help="eval batch size")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--q_reg", type=float, default=2e-2,
                    help="Learning rate for sparse projectors")
parser.add_argument("--d_reg", type=float, default=2e-2,
                    help="Learning rate for sparse projectors")
args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(model: D2SModel, train_dataloader, val_dataset, num_epochs, loss_fnc: BICELoss,  optimizer, scheduler, scaler, highest_recall_1):
    mask_ratio = torch.tensor(1.0)
    step = mask_ratio/(num_epochs*0.95)
    for epoch_idx, epoch in enumerate(range(0, num_epochs)):
        i = 0
        batch_loss = 0
        batch_rel_loss = 0
        batch_reg = 0
        q_len = []
        d_len = []
        for idx, batch in enumerate(tqdm(train_dataloader, desc="Training batch")):
            optimizer.zero_grad()
            batch_tokenized_texts, dense_texts, dense_imgs = batch
            batch_tokenized_texts = batch_tokenized_texts.to(device)
            dense_texts = dense_texts.to(device)
            dense_imgs = dense_imgs.to(device)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if torch.bernoulli(mask_ratio) == 1:
                    sparse_texts = model(
                        dense_texts, batch_tokenized_texts["input_ids"], batch_tokenized_texts["attention_mask"])
                else:
                    sparse_texts = model(dense_imgs)
                sparse_imgs = model(dense_imgs)
                rel_loss, reg = loss_fnc(
                    sparse_texts, sparse_imgs, dense_texts, dense_imgs)
                batch_rel_loss += rel_loss.item()
                batch_reg += reg
                loss = rel_loss + reg
                batch_loss += loss.item()
                q_len.append((sparse_texts > 0).float().sum(dim=-1).mean())
                d_len.append((sparse_imgs > 0).float().sum(dim=-1).mean())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            i += 1
        batch_loss = batch_loss / len(train_dataloader)
        batch_rel_loss = batch_rel_loss/len(train_dataloader)
        batch_reg = batch_reg / len(train_dataloader)
        recall1, recall5, recall10, mrr10, avg_flops = evaluate(
            model, val_dataset, vector_collator, mask_ratio=mask_ratio)
        if recall1 > highest_recall_1:
            highest_recall_1 = recall1
            print(f"Obtained higher recall@1: {highest_recall_1}")
            print(f"Saving checkpoint to {model_path}")
            model.save_pretrained(model_path)
        print(
            f"Epoch {epoch_idx+1} R@1 {recall1} R@5 {recall5} R@10 {recall10} loss {batch_loss} rel_loss {batch_rel_loss} reg {batch_reg} q_len {sum(q_len)/len(q_len)} d_len {sum(d_len)/len(d_len)} avg_flops {avg_flops}")
        mask_ratio = torch.relu(mask_ratio-step)
    return highest_recall_1


def evaluate(model, dataset, shared_collator, mask_ratio=torch.tensor(0.0), return_run_file=False, dense=False):
    text_collection, image_collection, qrels = dataset
    model.eval()
    img_dataloader = DataLoader(
        image_collection, batch_size=args.eval_batch_size, shuffle=False, num_workers=18, collate_fn=shared_collator)
    text_dataloader = DataLoader(
        text_collection, batch_size=args.eval_batch_size, shuffle=False, num_workers=18, collate_fn=shared_collator)
    all_text_embs = []
    text_ids = []
    all_image_embs = []
    image_ids = []
    for batch_texts in tqdm(text_dataloader, desc="Encoding texts"):
        text_ids.extend(batch_texts[0])
        if dense:
            all_text_embs.append(batch_texts[2].to(device))
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
                if torch.bernoulli(mask_ratio) == 1:
                    batch_text_embs = model(batch_texts[2].to(device), batch_texts[1]["input_ids"].to(
                        device), batch_texts[1]["attention_mask"].to(device))
                else:
                    batch_text_embs = model(
                        batch_texts[2].to(device)).to(device)
                all_text_embs.append(batch_text_embs)
    for batch_images in tqdm(img_dataloader, desc="Encoding images"):
        image_ids.extend(batch_images[0])
        if dense:
            all_image_embs.append(batch_images[1].to(device))
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
                batch_img_embs = model(batch_images[1].to(device))
                all_image_embs.append(batch_img_embs)
    scores = []
    flops = 0
    for batch_text in tqdm(all_text_embs, desc="Computing similarity scores"):
        batch_scores = []
        for batch_img in all_image_embs:
            flops += torch.sum((batch_img != 0).float().sum(dim=0) *
                               (batch_text != 0).float().sum(dim=0))
            batch_scores.append(batch_text.mm(batch_img.t()))
        batch_scores = torch.cat(batch_scores, dim=1)
        scores.append(batch_scores)
    del all_image_embs
    del all_text_embs
    torch.cuda.empty_cache()
    scores = torch.cat(scores, dim=0).to("cpu")
    sorted_indices = scores.argsort(dim=1, descending=True)
    run = defaultdict(OrderedDict)
    for i, txt_id in enumerate(tqdm(text_ids, desc="Creating run files")):
        for j in sorted_indices[i, :10]:
            run[txt_id][image_ids[j]] = scores[i][j].item()
    if qrels is None:
        metrics = {}
    else:
        metrics = ir_measures.calc_aggregate(
            [R@1, R@5, R@10, R@100, MRR@10], qrels, run)
    avg_flops = flops / scores.size(0) / scores.size(1)
    model.train()
    if return_run_file:
        return run, metrics[R@1], metrics[R@5], metrics[R@10], metrics[MRR@10], avg_flops
    else:
        return metrics[R@1], metrics[R@5], metrics[R@10], metrics[MRR@10], avg_flops


def prepare_data(dataset_repo):
    dense_embs = load_dataset(dataset_repo, data_files={"img_emb": "img_embs.parquet",
                                                        "text_emb": "text_embs.parquet"}, keep_in_memory=True).with_format("numpy")
    meta_data = json.load(open(hf_hub_download(
        repo_id=args.data, repo_type="dataset", filename="dataset_meta.json")))
    text_ids = dense_embs['text_emb']["id"]
    text_embs = dense_embs['text_emb']['emb']
    img_ids = dense_embs['img_emb']['id']
    img_embs = dense_embs['img_emb']['emb']
    txtid2row = dict(zip(text_ids, range(len(text_ids))))
    imgid2row = dict(zip(img_ids, range(len(img_ids))))

    train_image_ids = []
    train_captions = []
    train_caption_ids = []
    train_pairs = []

    val_image_ids = []
    val_captions = []
    val_caption_ids = []
    val_qrels = defaultdict(dict)

    test_image_ids = []
    test_captions = []
    test_caption_ids = []
    test_qrels = defaultdict(dict)

    for image in tqdm(meta_data['images'], desc="Processing meta data."):
        image_id = str(image["imgid"])
        caption_texts = [sent["raw"] for sent in image["sentences"]]
        caption_ids = [str(sent["sentid"]) for sent in image["sentences"]]
        if image["split"] == "train":
            train_image_ids.append(image_id)
            train_captions.extend(caption_texts)
            train_caption_ids.extend(caption_ids)
            train_pairs.extend([(sent_id, image_id)
                               for sent_id in caption_ids])
        if image['split'] == "val":
            val_image_ids.append(image_id)
            val_captions.extend(caption_texts)
            val_caption_ids.extend(caption_ids)
            for sent_id in caption_ids:
                val_qrels[sent_id][image_id] = 1
        if image['split'] == 'test':
            test_image_ids.append(image_id)
            test_captions.extend(caption_texts)
            test_caption_ids.extend(caption_ids)
            for sent_id in caption_ids:
                test_qrels[sent_id][image_id] = 1
    train_dataset = TrainDataset(
        dict(zip(train_caption_ids, train_captions)), txtid2row, imgid2row, text_embs, img_embs, train_pairs)
    val_text_collection = TextCollection(
        val_caption_ids, val_captions, txtid2row, text_embs)
    val_image_collection = ImageCollection(val_image_ids, imgid2row, img_embs)
    test_text_collection = TextCollection(
        test_caption_ids, test_captions, txtid2row, text_embs)
    test_image_collection = ImageCollection(
        test_image_ids, imgid2row, img_embs)
    return train_dataset, (val_text_collection, val_image_collection, val_qrels), (test_text_collection, test_image_collection, test_qrels)


if __name__ == "__main__":
    model = D2SModel()
    model.to(device)
    train_dataset, val_dataset, test_dataset = prepare_data(
        args.data)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vector_collator = VectorCollator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=18, collate_fn=vector_collator)
    temp = nn.Parameter(torch.tensor(
        args.temp, requires_grad=True, device=device))
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.vocab_layer_norm.parameters()) +
             list(model.vocab_projector.parameters()), "lr": 5e-5, "betas": (0.9, 0.999), "weight_decay": 0.0},
            {"params": list(model.proj.parameters()) + [temp], "lr": 1e-3}
        ],
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )
    num_training_steps = len(train_dataloader) * args.epochs
    num_warm_up = int(num_training_steps * 0.2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps)
    loss = BICELoss(temp=temp, q_reg=args.q_reg,
                    d_reg=args.d_reg, T=num_warm_up)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    test_dense_run, recall1, recall5, recall10, mrr10, dense_flops = evaluate(
        model, test_dataset, vector_collator, dense=True, return_run_file=True)
    print(
        f"dense performance: r@1: {recall1} r@5: {recall5} r@10: {recall10} mrr@10: {mrr10} dense_flops: {dense_flops}")
    model_dir = Path(
        f"output/{args.data}_qreg_{args.q_reg}_dreg_{args.d_reg}_tmp.tuned_{args.temp}")
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir/"model"
    highest_recall_1 = 0
    highest_recall_1 = train(model, train_dataloader, val_dataset,
                             args.epochs, loss, optimizer, scheduler, scaler, highest_recall_1)
    print("\nDone training")
    print(f"Loading best checkpoint from {model_path}")
    model = D2SModel.from_pretrained(model_path).to(device)
    mask_ratio = torch.tensor(0.0)
    print(f"Perform test evaluation.")
    test_sparse_run, test_r1, test_r5, test_r10, test_mrr10, avg_flops = evaluate(
        model, test_dataset, vector_collator, return_run_file=True, mask_ratio=mask_ratio)
    run_file_path = model_dir/"test_run_file.trec"
    write_trec_file(test_sparse_run, run_file_path)
    result_path = model_dir/"test_result.txt"
    print(f"Saving test metrics to {result_path}")
    corr_res = cal_correaltion(
        test_sparse_run, test_dense_run, test_dataset[2])
    with open(result_path, "w") as f:
        f.write(f"R@1: {test_r1}\n")
        f.write(f"R@5: {test_r5}\n")
        f.write(f"R@10: {test_r10}\n")
        f.write(f"MRR@10: {test_mrr10}\n")
        f.write(f"FLOPS: {avg_flops}\n")
    print(
        f"Test r1: {test_r1}, r5: {test_r5}, r10: {test_r10} mrr@10: {test_mrr10} flops: {avg_flops}")
    corr_path = model_dir/"test_correlations.json"
    with open(corr_path, "w") as f:
        json.dump(corr_res, f)
