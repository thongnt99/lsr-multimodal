from datasets import load_dataset
from huggingface_hub import hf_hub_download
from collections import defaultdict
import torch
from torch import nn
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.optimization import get_linear_schedule_with_warmup
import ir_measures
from ir_measures import *
from collections import OrderedDict
import torch.nn.functional as F
from regularizer import *
from dataset import *
from model import MLM
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
parser.add_argument("--model", type=str, default="blip",
                    help="Selecting models")
parser.add_argument("--q_reg", type=float, default=2e-2,
                    help="Learning rate for sparse projectors")
parser.add_argument("--d_reg", type=float, default=2e-2,
                    help="Learning rate for sparse projectors")
parser.add_argument("--mask_ratio", type=float, default=-1, help="mask ratio")
args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate(model, text_collection, image_collection, qrels, shared_collator, mask_ratio=torch.tensor(0.0), return_run_file=False, dense=False):
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
            all_text_embs.append(batch_texts[2].to("cuda"))
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
                if torch.bernoulli(mask_ratio) == 1:
                    batch_text_embs = model(batch_texts[2].to("cuda"), batch_texts[1]["input_ids"].to(
                        "cuda"), batch_texts[1]["attention_mask"].to("cuda"))
                else:
                    batch_text_embs = model(
                        batch_texts[2].to("cuda")).to("cuda")
                all_text_embs.append(batch_text_embs)
    for batch_images in tqdm(img_dataloader, desc="Encoding images"):
        image_ids.extend(batch_images[0])
        if dense:
            all_image_embs.append(batch_images[1].to("cuda"))
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
                batch_img_embs = model(batch_images[1].to("cuda"))
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


if __name__ == "__main__":
    dense_embs = load_dataset(args.data, data_files={"img_emb": "img_embs.parquet",
                                                     "text_emb": "text_embs.parquet"}, keep_in_memory=True).with_format("numpy")
    meta_data = json.load(open(hf_hub_download(
        repo_id=args.data, repo_type="dataset", filename="dataset_meta.json")))
    text_ids = dense_embs['text_emb']["id"]
    text_embs = dense_embs['text_emb']['emb']
    img_ids = dense_embs['img_emb']['id']
    img_embs = dense_embs['img_emb']['emb']
    txtid2row = dict(zip(text_ids, range(len(text_ids))))
    imgid2row = dict(zip(img_ids, range(len(img_ids))))
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vector_collator = VectorCollator(tokenizer)

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

    files = []
    test_files = []
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
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=18, collate_fn=vector_collator)
    val_text_collection = TextCollection(
        val_caption_ids, val_captions, txtid2row, text_embs)
    val_image_collection = ImageCollection(val_image_ids, imgid2row, img_embs)
    test_text_collection = TextCollection(
        test_caption_ids, test_captions, txtid2row, text_embs)
    test_image_collection = ImageCollection(
        test_image_ids, imgid2row, img_embs)
    model = MLM(256)
    model.to("cuda")
    pretrained_layers = list(model.vocab_layer_norm.parameters()) + \
        list(model.vocab_projector.parameters())
    optimizer1 = torch.optim.AdamW(
        pretrained_layers, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    temp = nn.Parameter(torch.tensor(
        args.temp, requires_grad=True, device="cuda"))
    new_layers = list(model.proj.parameters()) + [temp]
    optimizer2 = torch.optim.AdamW(
        new_layers, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warm_up = int(num_training_steps * 0.2)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps)
    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps)
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    q_regularizer = L1(weight=args.q_reg, T=num_warm_up)
    d_regularizer = L1(weight=args.d_reg, T=num_warm_up)
    test_dense_run, recall1, recall5, recall10, mrr10, dense_flops = evaluate(
        model, test_text_collection, test_image_collection, test_qrels, vector_collator, dense=True, return_run_file=True)
    print(
        f"dense performance: r@1: {recall1} r@5: {recall5} r@10: {recall10} mrr@10: {mrr10} dense_flops: {dense_flops}")
    eval_metrics = [[], [], []]
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    model_dir = Path(
        f"output/{args.data}_qreg_{args.q_reg}_dreg_{args.d_reg}_tmp.tuned_{args.temp}")
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir/"model.pt"

    def train(num_epochs, highest_recall_1, masking=True):
        if args.mask_ratio >= 0:
            mask_ratio = torch.tensor(args.mask_ratio)
        else:
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
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                batch_tokenized_texts, batch_txt_embs, batch_img_embs = batch
                batch_tokenized_texts = batch_tokenized_texts.to("cuda")
                batch_txt_embs = batch_txt_embs.to("cuda")
                batch_img_embs = batch_img_embs.to("cuda")
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    if torch.bernoulli(mask_ratio) == 1:
                        sparse_texts = model(
                            batch_txt_embs, batch_tokenized_texts["input_ids"], batch_tokenized_texts["attention_mask"])
                    else:
                        sparse_texts = model(batch_txt_embs)

                    sparse_imgs = model(batch_img_embs)
                    logits_per_image = sparse_imgs @ sparse_texts.t()
                    logits_per_text = logits_per_image.t()
                    with torch.no_grad():
                        scores_dense_i2t = batch_img_embs @ batch_txt_embs.t()
                        prob_dense_i2t = torch.softmax(
                            scores_dense_i2t/temp, dim=1)
                        prob_dense_t2i = torch.softmax(
                            scores_dense_i2t.t()/temp, dim=1)
                    loss = (loss_img(logits_per_image, prob_dense_i2t) +
                            loss_txt(logits_per_text, prob_dense_t2i))/2
                    reg = (q_regularizer(sparse_texts) +
                           d_regularizer(sparse_imgs))/2
                    batch_rel_loss += loss.item()
                    batch_reg += reg
                    loss += reg
                    batch_loss += loss.item()
                    q_len.append((sparse_texts > 0).float().sum(dim=-1).mean())
                    d_len.append((sparse_imgs > 0).float().sum(dim=-1).mean())
                scaler.scale(loss).backward()
                scaler.step(optimizer1)
                scaler.step(optimizer2)
                scaler.update()
                scheduler1.step()
                scheduler2.step()
                q_regularizer.step()
                d_regularizer.step()
                i += 1
            batch_loss = batch_loss / len(train_dataloader)
            batch_rel_loss = batch_rel_loss/len(train_dataloader)
            batch_reg = batch_reg / len(train_dataloader)
            losses.append(batch_loss)
            recall1, recall5, recall10, mrr10, avg_flops = evaluate(
                model, val_text_collection, val_image_collection, val_qrels, vector_collator, mask_ratio=mask_ratio)
            if recall1 > highest_recall_1:
                highest_recall_1 = recall1
                print(f"Obtained higher recall@1: {highest_recall_1}")
                print(f"Saving checkpoint to {model_path}")
                torch.save(model.state_dict(), model_path)

            print(
                f"Epoch {epoch_idx+1} R@1 {recall1} R@5 {recall5} R@10 {recall10} loss {batch_loss} rel_loss {batch_rel_loss} reg {batch_reg} q_len {sum(q_len)/len(q_len)} d_len {sum(d_len)/len(d_len)} avg_flops {avg_flops}")
            if args.mask_ratio < 0:
                mask_ratio = torch.relu(mask_ratio-step)
            eval_metrics[0].append(recall1)
            eval_metrics[1].append(recall5)
            eval_metrics[2].append(recall10)
        return highest_recall_1
    highest_recall_1 = 0
    highest_recall_1 = train(args.epochs, highest_recall_1,
                             masking=False)
    print("")
    print("Done training")
    print(losses)
    print("R@1: ", eval_metrics[0])
    print("R@5: ", eval_metrics[1])
    print("R@10: ", eval_metrics[2])
    print(
        f"Loading best checkpoint from {model_path}")
    model.load_state_dict(torch.load(model_path))
    if args.mask_ratio >= 0:
        mask_ratio = torch.tensor(args.mask_ratio)
    else:
        mask_ratio = torch.tensor(0.0)
    print(f"Perform test evaluation with mask ratio = {args.mask_ratio}")

    test_sparse_run, test_r1, test_r5, test_r10, test_mrr10, avg_flops = evaluate(
        model, test_text_collection, test_image_collection, test_qrels, vector_collator, return_run_file=True, mask_ratio=mask_ratio)
    run_file_path = model_dir/"test_run_file.trec"
    print(f"Saving test run file to {run_file_path}")
    write_trec_file(test_sparse_run, run_file_path)
    result_path = model_dir/"test_result.txt"
    print(f"Saving test metrics to {result_path}")
    corr_res = cal_correaltion(test_sparse_run, test_dense_run, test_qrels)
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
