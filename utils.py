import numpy as np
import scipy.stats as stats
import ir_measures
from ir_measures import *


def cal_correaltion(run1, run2, qrels=None):
    taus = {}
    for qid in run1:
        doc_id_set = list(
            set(run1[qid].keys()).intersection(run2[qid].keys()))
        rank1 = np.array([run1[qid][did] for did in doc_id_set])
        rank2 = np.array([run2[qid][did] for did in doc_id_set])
        taus[qid] = stats.kendalltau(rank1, rank2)

    avg_tau = np.nanmean(np.array(list(taus.values())))
    res = {"avg_tau": avg_tau}
    if qrels:
        run1_r1 = {}
        run1_r5 = {}
        run1_r10 = {}
        run1_mrr10 = {}
        run2_r1 = {}
        run2_r5 = {}
        run2_r10 = {}
        run2_mrr10 = {}
        for metric in ir_measures.iter_calc([R@1, R@5, R@10, MRR@10], qrels, run1):
            if metric.measure == R@1:
                run1_r1[metric.query_id] = metric.value
            elif metric.measure == R@5:
                run1_r5[metric.query_id] = metric.value
            elif metric.measure == R@10:
                run1_r10[metric.query_id] = metric.value
            elif metric.measure == MRR@10:
                run1_mrr10[metric.query_id] = metric.value
        for metric in ir_measures.iter_calc([R@1, R@5, R@10,  MRR@10], qrels, run2):
            if metric.measure == R@1:
                run2_r1[metric.query_id] = metric.value
            elif metric.measure == R@5:
                run2_r5[metric.query_id] = metric.value
            elif metric.measure == R@10:
                run2_r10[metric.query_id] = metric.value
            elif metric.measure == MRR@10:
                run2_mrr10[metric.query_id] = metric.value

        def cal_pearson(metrics1, metrics2):
            list1 = []
            list2 = []
            for qid in metrics1:
                list1.append(metrics1[qid])
                list2.append(metrics2[qid])
            corr = stats.pearsonr(list1, list2)
            return corr
        res["pearsonr@r1"] = cal_pearson(run1_r1, run2_r1)
        res["pearsonr@r5"] = cal_pearson(run1_r5, run2_r5)
        res["pearsonr@r10"] = cal_pearson(run1_r10, run2_r10)
        res["pearsonr@mrr10"] = cal_pearson(run1_mrr10, run2_mrr10)
    res["taus"] = taus
    return res


def write_trec_file(trec_run, run_file_path):
    print(f"Saving TREC run file to {run_file_path}")
    with open(run_file_path, "w") as f:
        for qid in trec_run:
            for rank, did in enumerate(trec_run[qid]):
                f.write(
                    f"{qid} Q0 {did} {rank+1} {trec_run[qid][did]} lsr42\n")
