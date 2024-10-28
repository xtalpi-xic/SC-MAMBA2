# %%
import copy
import gc
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from pathlib import Path
import sys
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings

import torch
import scanpy as sc
import scvi
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

import scmamba
from scmamba.model import MambaModel
from scmamba import prepare_data, prepare_dataloader, evaluate, eval_testdata, train
from scmamba.tokenizer import tokenize_and_pad_batch
from scmamba.tokenizer.gene_tokenizer import GeneVocab
from scmamba.loss import masked_mse_loss
from scmamba.preprocess import Preprocessor
from scmamba.utils import set_seed, load_pretrained

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Step1: Specify hyper-parameter setup for integration task
# Here we provide some hyper-parameter recommendations here for the integration task. Note that the PBMC 10K dataset contains multiple batches to be integrated. Therefore, in addition to the default gene modelling objectives, we also turn on ESC, DAR and DSBN objectives specifically to faciliate batch integration.

# %%
config = SimpleNamespace(
    task = 'integration',
    seed=42,
    dataset_name="PBMC_10K", # Dataset name
    load_model = '/mnt/hn19storage/fan.zhang/model_save/whole_body_2023_pretrain_1200_1536_36',
    embsize=1536,
    mamba_layer=36,
    nlayers=36,
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0, # DAR objective weight for batch correction
    mask_ratio=0.4, # Default mask ratio
    epochs=30, # Default number of epochs for fine-tuning
    n_bins=51, # Default number of bins for value binning in data pre-processing
    lr=5e-5, # Default learning rate for fine-tuning
    batch_size=1, # Default batch size for fine-tuning
    dropout=0, # Default dropout rate during model fine-tuning
    schedule_interval = 3,
    schedule_ratio=0.9,  # Default rate for learning rate decay
    save_eval_interval=1, # Default model evaluation 
    log_interval=200, # Default log interval
    pre_norm=False, # Default setting
    amp=True,  # # Default setting: Automatic Mixed Precision
    mask_value = -1,
    pad_value = -2,
    n_hvg = 1200,
    max_seq_len = 1201,
    per_seq_batch_sample = True,
    GEP=False,
    GEPC=True,  # Gene expression modelling for cell objective
    CLS=False,
    DAR=True,
    DSBN=True,
    use_batch_labels=True,
    explicit_zero_prob=True,
    use_mod=False,
    include_zero_gene=True,
    pad_token = "<pad>",
    special_tokens = ["<pad>", "<cls>", "<eoc>"],
    input_layer_key="X_binned"
)
pad_token = "<pad>"
pad_value = -2
special_tokens = [pad_token, "<cls>", "<eoc>"]
input_layer_key="X_binned"


set_seed(config.seed)

# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/intergtation_2023wholebody_epoch6_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scmamba.logger
scmamba.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"save to {save_dir}")
logger.info(config)

# %%
if dataset_name == "PBMC_10K":
    adata = scvi.data.pbmc_dataset(save_path = './data')  # 11990 × 3346
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    data_is_raw = True

# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


# %%
# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=config.n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)


# %%
if config.per_seq_batch_sample:
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# %%
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
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
    
    # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will be overriden by the "
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
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


# %% [markdown]
#  ## Step 3: Load the pre-trained scGPT model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
model = MambaModel(
    ntokens,
    embsize,
    mamba_layer,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file), verbose=False)

model.to(device)
logger.info(model)

# %%
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.schedule_interval, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)



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
        f"valid loss/mse {val_loss:5.6f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        # eval on testdata
        results = eval_testdata(
            best_model,
            adata_t=adata_sorted if config.per_seq_batch_sample else adata,
            gene_ids=gene_ids,
            vocab=vocab,
            config=config,
            logger=logger,
            include_types=["cls"]
        )
        results["batch_umap"].savefig(
            save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
        )

        results["celltype_umap"].savefig(
            save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
        )
        metrics_to_log = {"test/" + k: v for k, v in results.items()}
        metrics_to_log["test/best_model_epoch"] = best_model_epoch
        logger.info(metrics_to_log)
        logger.info({"avg_bio": results.get("avg_bio", 0.0)})

    scheduler.step()

# %%
# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")

# %%

glob_str = os.path.join(save_dir, "best_model.pt")
gc.collect()


