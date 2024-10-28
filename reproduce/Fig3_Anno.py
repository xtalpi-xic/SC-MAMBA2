# %%
import copy
import gc
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from pathlib import Path
import shutil
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import pickle
import torch
import scanpy as sc
import seaborn as sns
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix
from types import SimpleNamespace

import scmamba
from scmamba.model import MambaModel
from scmamba import prepare_data, prepare_dataloader, evaluate, eval_testdata, train
from scmamba.tokenizer import tokenize_and_pad_batch
from scmamba.tokenizer.gene_tokenizer import GeneVocab
from scmamba.loss import masked_mse_loss
from scmamba.preprocess import Preprocessor
from scmamba.utils import set_seed

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Step1: Specify hyper-parameter setup for cell-type annotation task
# Listed below are some hyper-parameter recommendations for the cell-type task. Note that the CLS objective is on to facilitate cell-type classification.

# %%
config = SimpleNamespace(
    task = 'annotation',
    seed=0,
    dataset_name="pancreas_10k",
    load_model='/mnt/hn19storage/fan.zhang/model_save/whole_body_2023_pretrain_1200_1536_36/',
    embsize=1536,
    mamba_layer=36,
    nlayers=36,
    ecs_thres=0.0,
    dab_weight=0.0,
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    lr=1e-4,
    batch_size=1,
    dropout=0.5,
    layer_size=1536,
    nlayers=4,
    MVC=False,
    schedule_ratio=0.5,
    save_eval_interval=5,
    max_seq_len = 3001,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,
    even_binning=True,
    input_emb_style ="continuous",
    pad_token="<pad>",
    special_tokens=["<pad>", "<cls>", "<eoc>"],
    mask_value="auto",
    input_style="binned",  # "normed_raw", "log1p", or "binned"
    output_style="binned",  # "normed_raw", "log1p", or "binned"
    CLS=True,
    ECS=False,
    DAB=False,
    DSMB=False,
    input_batch_labels=False,
    cell_emb_style="cls",
    do_sample_in_train=False,
    per_seq_batch_sample=False,
    schedule_interval=2,
    log_interval=100,
    do_eval_scib_metrics=True,
    pad_value=-2,
    n_input_bins=lambda: config.n_bins
)
pad_token = "<pad>"
pad_value = -2
special_tokens = [pad_token, "<cls>", "<eoc>"]
input_layer_key="X_binned"


set_seed(config.seed)

# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/anno_2023wholebody_cls_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scmamba.logger
scmamba.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"save to {save_dir}")
logger.info(config)

# %%
if dataset_name == "ms":
    data_dir = Path("/mnt/hn19storage/fan.zhang/scmamba_finetunning/tutorials/data/ms")
    adata = sc.read(data_dir / "c_data.h5ad")
    adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
elif dataset_name == "pancreas_10k":
    data_dir = Path("/mnt/hn19storage/fan.zhang/scmamba_finetunning/tutorials/data/annotation_pancreas")
    adata = sc.read(data_dir / "demo_train.h5ad")
    adata_test = sc.read(data_dir / "demo_test.h5ad")
    adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
    adata.obs["str_batch"] = "train"
    adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
    adata_test.obs["str_batch"] = "test"
    # merge the two datasets
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    data_is_raw = True
    filter_gene_by_counts = False
elif dataset_name == "myeloid":
    data_dir = Path("/mnt/hn19storage/fan.zhang/scmamba_finetunning/tutorials/data/mye")
    adata = sc.read(data_dir / "reference_adata.h5ad")
    adata_test = sc.read(data_dir / "query_adata.h5ad")
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"
    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
                
# make the batch category column
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=filter_gene_by_counts,
    filter_cell_by_counts=False,
    log1p=data_is_raw,
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,
    result_binned_key="X_binned",
    even_binning=config.even_binning,
)

# Split adata into train and test based on 'str_batch'
adata_test = adata[adata.obs["str_batch"] == "1"].copy()
adata = adata[adata.obs["str_batch"] == "0"].copy()

# Apply preprocessing to both train and test sets
preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

# Extract layers and other information after preprocessing
input_layer_key = {
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[config.input_style]

all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype_id"].tolist()
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

# %%
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    if config.task in ['integration', 'annotation']:
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
    
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        mamba_layer = model_configs["nlayers"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        embsize = config.layer_size 
        mamba_layer = config.mamba_layer
        nlayers = config.nlayers  
        d_hid = config.layer_size
        old_vocab = vocab
else:
    embsize = config.layer_size 
    mamba_layer = config.mamba_layer
    nlayers = config.nlayers  
    d_hid = config.layer_size

vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

# %%
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=config.include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\t feature length: {tokenized_valid['genes'].shape[1]}"
)

# %% [markdown]
# ## Step 3: Load the pre-trained SC-MAMBA2 model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
model = MambaModel(
    ntokens,
    embsize,
    mamba_layer,
    d_hid,
    nlayers,
    n_cls=num_types,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    use_batch_labels=config.input_batch_labels,
    num_batch_labels=num_batch_types,
    n_input_bins=config.n_input_bins,
    ecs_threshold=config.ecs_thres,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)
logger.info(model)

# %%
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.schedule_interval, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# %%
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp,  dtype=torch.bfloat16):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.input_batch_labels or config.DSBN else None,
                CLS=config.CLS,
                MVC=config.MVC,
                ECS=config.ECS,
                do_sample=config.do_sample_in_train,
            )
            
            loss = 0.0
            metrics_to_log = {}
            loss = criterion_cls(output_dict["cls_output"], celltype_labels)
            metrics_to_log.update({"train/cls": loss.item()})
            error_rate = 1 - (
                (output_dict["cls_output"].argmax(1) == celltype_labels)
                .sum()
                .item()
            ) / celltype_labels.size(0)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        logger.info(metrics_to_log)
        total_loss += loss.item()
        total_error += error_rate
        
        if batch % config.log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / config.log_interval
            cur_loss = total_loss / config.log_interval
            cur_error = total_error / config.log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"err {cur_error:5.2f} | " if config.CLS else "")
            )
            total_loss = 0
            total_error = 0
            start_time = time.time()

def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp,  dtype=torch.bfloat16):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.input_batch_labels or config.DSBN else None,
                    CLS=config.CLS,
                    MVC=config.MVC,
                    ECS=config.ECS,
                    do_sample=config.do_sample_in_train,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    logger.info(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


# %% [markdown]
# ## Step 4: Finetune SC-MAMBA2 with task-specific objectives

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None


for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(
        tokenized_train=tokenized_train, 
        tokenized_valid=tokenized_valid, 
        train_batch_labels=train_batch_labels,
        valid_batch_labels=valid_batch_labels,
        config=config,
        epoch=epoch,
        train_celltype_labels=valid_celltype_labels,
        valid_celltype_labels=valid_celltype_labels,
        sort_seq_batch=config.per_seq_batch_sample)
    
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    train(
        model=model,
        loader=train_loader,
        vocab=vocab,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        epoch=epoch,
    )
    
    val_loss = evaluate(
        model=model,
        loader=valid_loader,
        vocab=vocab,
        device=device,
        config=config,
        logger=logger,
        epoch=epoch
    )
    
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with val loss {best_val_loss:5.4f}")

    scheduler.step()



# %% [markdown]
# ## Step 5: Inference with fine-tuned SC-MAMBA2 model
# In the cell-type annotation task, the fine-tuned SC-MAMBA2 predicts cell-type labels for query set as inference. The model performance is evaluated on standard classificaton metrics. Here we visualize the predicted labels over the SC-MAMBA2 cell embeddings, and present the confusion matrix for detailed classification performance on the cell-group level.

# %%
predictions, labels, results = test(best_model, adata_test, gene_ids, vocab, config, device, logger)
adata_test.obs["predictions"] = [id2type[p] for p in predictions]

# plot
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] 
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
    sc.pl.umap(
        adata_test,
        color=["celltype", "predictions"],
        palette=palette_,
        show=False,
        wspace=0.5,
    )
    plt.savefig(save_dir / "results.png", dpi=300)

save_dict = {
    "predictions": predictions,
    "labels": labels,
    "results": results,
    "id_maps": id2type
}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)

logger.info(results)

# %%
from sklearn.metrics import confusion_matrix
celltypes = list(celltypes)
for i in set([id2type[p] for p in predictions]):
    if i not in celltypes:
        celltypes.remove(i)
cm = confusion_matrix(labels, predictions)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

# %%
# save the model into the save_dir
torch.save(best_model.state_dict(), save_dir / "model.pt")
