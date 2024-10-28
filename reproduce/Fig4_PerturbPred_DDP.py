# %%
import json
import os
import sys
import csv
import time
import copy
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from types import SimpleNamespace

from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

import scmamba
from scmamba.model import MambaGenerator
from scmamba.loss import masked_mse_loss
from scmamba.tokenizer.gene_tokenizer import GeneVocab
from scmamba.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")


# %%
local_rank = int(os.getenv('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl', rank=local_rank)
device = torch.device("cuda")
world_size = torch.distributed.get_world_size()
torch.distributed.barrier()

set_seed(42)

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 6000

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
amp = True
load_model = "/mnt/hn19storage/fan.zhang/model_save/whole_body_2023_pretrain_1200_1536_36/"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 16
eval_batch_size = 16
epochs = 30
schedule_interval = 1
early_stop = 40
test_interval = 5
# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 24  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability

# logging
log_interval = 100

# dataset and evaluation choices
data_name = "norman"
split = "simulation"
perts_to_plot = ["SAMD1+ZBTB1"]


save_dir = Path(f"./save/perturb_mamba_2023wholebody_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
if local_rank in [0, -1]:
    print(f"saving to {save_dir}")

logger = scmamba.logger
scmamba.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
if local_rank in [0, -1]:
    logger.info(f"gpu: {world_size} max_seq_len: {max_seq_len} lr: {lr} bs: {batch_size} eval_bs: {eval_batch_size} epoch: {epochs} dropout: {dropout}")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


pert_data = PertData("./data/")
# pert_data.load(data_name=data_name,data_path='./data/changed_tk2021/')
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)


if load_model is not None:
    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    if local_rank in [0, -1]:
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
    genes = pert_data.adata.var["gene_name"].tolist()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if local_rank in [0, -1]:
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
)  # vocab["<pad>"] 为 60530
n_genes = len(genes)

#  # Create and train scGpt

ntokens = len(vocab)  # size of vocabulary
model = MambaGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
)
if load_param_prefixs is not None and load_model is not None:
    # only load params that start with the prefix
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        if local_rank in [0, -1]:
            logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
elif load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        if local_rank in [0, -1]:
            logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            if local_rank in [0, -1]:
                logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
if local_rank in [0, -1]:
    logger.info(model)

model.to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=amp)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)
        
        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp, dtype=torch.bfloat16):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            if local_rank in [0, -1]:
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.4f} | mse {cur_mse:5.4f} |"
                )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    model.to(device)
    pert_cat, pred, truth, pred_de, truth_de =[], [], [], [], []
    results = {}
    for itr, batch in enumerate(val_loader):
        batch.to(device)
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            batch_size = len(batch.pert)
            x: torch.Tensor = batch.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, -1)
            pert_flags = x[:, 1].long().view(batch_size, -1)
            target_gene_values = batch.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                assert gene_ids is not None
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(ori_gene_values.size(1),device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )
                
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )

                with torch.cuda.amp.autocast(enabled=amp, dtype=torch.bfloat16):
                    output_dict = model(
                        mapped_input_gene_ids,
                        input_values,
                        input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=CLS,
                        CCE=CCE,
                        MVC=MVC,
                        ECS=ECS,
                        do_sample=True,
                    )
                output_values = output_dict["mlm_output"].float()
                pred_gene_values = torch.zeros_like(ori_gene_values)
                pred_gene_values[:, input_gene_ids] = output_values

            pred.extend(pred_gene_values.cpu())
            truth.extend(target_gene_values.cpu())
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(pred_gene_values[itr, de_idx])
                truth_de.append(target_gene_values[itr, de_idx])
                
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float32)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float32)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float32)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float32)
    return results

def predict(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred

def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None
):

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    if query.split("+")[1] == "ctrl":
        pred = predict(model, [[query.split("+")[0]]], pool_size=pool_size)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, [query.split("+")], pool_size=pool_size)
        pred = pred["_".join(query.split("+"))][de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    plt.figure(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        plt.savefig(save_file, bbox_inches="tight", transparent=False)
    # plt.show()


best_val_loss = float("inf")
best_val_corr = 0
best_model = None
patience = 0

train_loader = pert_data.dataloader["train_loader"]
valid_loader = pert_data.dataloader["val_loader"]
test_loader = pert_data.dataloader["test_loader"]
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    train(
        model,
        train_loader,
    )
    
    val_res = evaluate(model.module, valid_loader)
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    if local_rank in [0, -1]:
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

    if epoch % test_interval == 0 or epoch == epochs:
        best_model = copy.deepcopy(model)
        test_res = evaluate(best_model.module, test_loader)
        if local_rank in [0, -1]:
            # torch.save(best_model.module.state_dict(), save_dir / "best_model.pt")
            # predict(best_model, [["FEV"], ["FEV", "SAMD11"]])
            for p in perts_to_plot:
                plot_perturbation(best_model.module, p, pool_size=300, save_file=f"{save_dir}/{p}.png")

            ##### Run test value


            test_metrics = compute_perturbation_metrics(
                test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
            )
            print(test_metrics)

            # save the dicts in json
            with open(f"{save_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)

            deeper_res = deeper_analysis(pert_data.adata, test_res)
            non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

            metrics = ["pearson_delta", "pearson_delta_de"]
            metrics_non_dropout = [
                "pearson_delta_top20_de_non_dropout",
                "pearson_top20_de_non_dropout",
            ]
            subgroup_analysis = {}
            for name in pert_data.subgroup["test_subgroup"].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in pert_data.subgroup["test_subgroup"].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(deeper_res[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(non_dropout_res[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    logger.info("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))
            
            overall_metrics = {m: [] for m in metrics + metrics_non_dropout}
            for name, result in subgroup_analysis.items():
                for m, value in result.items():
                    overall_metrics[m].append(value)
            for m, values in overall_metrics.items():
                overall_avg = np.mean(values)
                logger.info("overall_average_" + m + ": " + str(overall_avg))
    
    scheduler.step()




