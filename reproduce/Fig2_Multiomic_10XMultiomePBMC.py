# %%
import copy
import gc
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path
import time
import warnings
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import pandas as pd
import episcanpy as epi
from types import SimpleNamespace

import scmamba
from scmamba import prepare_data, prepare_dataloader, evaluate, eval_testdata, train
from scmamba.tokenizer import tokenize_and_pad_batch
from scmamba.model import MambaModel
from scmamba.tokenizer.gene_tokenizer import GeneVocab
from scmamba.loss import masked_mse_loss
from scmamba.preprocess import Preprocessor
from scmamba.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Step1: Specify hyper-parameter setup for integration task
# Here we provide some hyper-parameter recommendations for the multiomic integration task. Note that the BMMC dataset contains multiple batches to be integrated. Therefore, in addition to the default gene modelling objectives, we also turn on DAR objectives specifically to faciliate batch integration. We also turn on the use_mod argument as default to ensure that the model is modality-aware during training.

# %%
config = SimpleNamespace(
    task = 'multiomic',
    seed=2,
    dataset_name="10x-Multiome-Pbmc10k", # Dataset name
    do_train=True, # Flag to indicate whether to do update model parameters during training
    load_model = '/mnt/hn19storage/fan.zhang/model_save/whole_body_2023_pretrain_1200_1536_36',
    #"../save/scGPT_human", # Path to pre-trained model
    freeze = False, #freeze
    GEP=True, # Gene expression modelling
    GEPC=True, # Gene expression modelling for cell objective
    CLS=False,
    ESC=False,
    DAR = False, # DAR objective weight for batch correction
    DSBN = False,  # Domain-spec batchnorm,
    mask_ratio=0.4, # Default mask ratio
    explicit_zero_prob = False,  # whether explicit bernoulli for zeros
    ecs_thres=0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    use_batch_labels = False,
    use_mod = True,
    per_seq_batch_sample = False,
    epochs=45, # Default number of epochs for fine-tuning
    input_layer_key = "X_binned", # Default expression value binning in data pre-processing
    n_bins=51, # Default number of bins for value binning in data pre-processing
    n_hvg = 1200,  # Default number of highly variable genes
    n_hvp = 4000,
    max_seq_len = 5202, # Default n_hvg+1
    lr=5e-4, # Default learning rate for fine-tuning
    batch_size=16, # Default batch size for fine-tuning
    layer_size=1536,
    nlayers=4,
    mamba_layer=36,
    dropout=0.2, # Default dropout rate during model fine-tuning
    schedule_interval = 1, # Default interval for learning rate decay
    schedule_ratio=0.95,  # Default rate for learning rate decay
    save_eval_interval=5, # Default model evaluation interval
    log_interval=100, # Default log interval
    pre_norm=False, # Default setting
    amp=True,  # Default setting: Automatic Mixed Precision
    pad_token = "<pad>",
    mask_value = -1,
    pad_value = -2,
    include_zero_gene = True,
    even_binning = False,
)

set_seed(config.seed)

# %%
# settings for input and preprocessing
special_tokens = [config.pad_token, "<cls>", "<eoc>"]

# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_mamba_multiomics_2023wholebody_epoch6_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"save to {save_dir}")
logger = scmamba.logger
scmamba.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(config)

# %% [markdown]
# ## Step 2: Load and pre-process data
if dataset_name == '10x-Multiome-Pbmc10k':
    adata = sc.read(
        "/mnt/hn19storage/fan.zhang/scmamba_finetunning/data/10x-Multiome-Pbmc10k-RNA.h5ad"
    )
    adata.obs["celltype"] = adata.obs["cell_type"].astype(str).astype('category')
    adata_atac = sc.read(
        "/mnt/hn19storage/fan.zhang/scmamba_finetunning/data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    )
    data_is_raw = True
    adata.obs["batch_id"] =  np.repeat(0, len(adata.obs["celltype"]))
    adata.obs["str_batch"] = np.array([str(i) for i in np.random.randint(2, size=len(adata.obs["celltype"]))])
    adata.obs["str_batch"] = adata.obs["str_batch"].astype('category')
    adata.var["gene_name"] = adata.var.index.tolist()

if config.use_mod:
    gene_rna_df = adata.var.loc[:, ['chrom', 'chromStart', 'chromEnd']].copy()
    gene_rna_df['mod'] = 'RNA'
    gene_atac_df = adata_atac.var.loc[:, ['chrom', 'chromStart', 'chromEnd']].copy()
    gene_atac_df['mod'] = 'ATAC'
    gene_loc_df = pd.concat([gene_rna_df, gene_atac_df])
    gene_loc_df['mod'] = gene_loc_df['mod'].astype('category')
# %% [markdown]
# ### 2.3 Pre-process the data
# We follow the standardized pipline of depth normalization, log normalization, and highly vairable gene (HVG) selection for data pre-processing. We further introduce value binning to obtain the relative expressions of each HVG. Given multiple sequencing modalities, we perform the pre-processing steps on each individual modality first, and then combine them into multi-modal sequences as model input.

# %% [markdown]
# #### 2.3.1 Pre-process the RNA data
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=1,  # step 1
    filter_cell_by_counts=1,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=config.n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    even_binning=config.even_binning,
)
preprocessor(adata, batch_key=None)
# %% [markdown]
# #### 2.3.2 Pre-process the Protein data

epi.pp.filter_cells(adata_atac, min_features = 1)
epi.pp.filter_features(adata_atac, min_cells = 1)
epi.pp.cal_var(adata_atac)
epi.pp.variability_features(adata_atac, nb_features=config.n_hvp, show=False)
adata_atac.raw = adata_atac
adata_atac = epi.pp.select_var_feature(adata_atac, nb_features=config.n_hvp, show=False, copy=True)

def digitize(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    assert x.ndim == 1 and bins.ndim == 1
    left_digits = np.digitize(x, bins)
    right_difits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))  # uniform random numbers
    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def do_binning(layer_data):
    binned_rows = []
    bin_edges = []
    for row in layer_data:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, config.n_bins - 1))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        if config.even_binning:
            non_zero_digits = digitize(non_zero_row, bins)
        else:
            non_zero_digits = np.digitize(non_zero_row, bins)
        assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= config.n_bins - 1
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        binned_rows.append(binned_row)
        bin_edges.append(np.concatenate([[0], bins]))
    return np.stack(binned_rows)

layer_data = np.array(adata_atac.X.todense())
adata_atac.obsm["atac_expression_binned"] = do_binning(layer_data)

# %% [markdown]
# #### 2.3.3 Combine RNA, and Protein data

append_names = adata_atac.var.index.tolist()
adata = AnnData(
    X=np.concatenate(
        [adata.layers["X_binned"], adata_atac.obsm["atac_expression_binned"]], axis=1
        ),
        obs=adata.obs,
        var=pd.DataFrame(index=adata.var.gene_name.tolist() + append_names),
        layers={
            "X_binned": np.concatenate(
                [adata.layers["X_binned"], adata_atac.obsm["atac_expression_binned"]], axis=1
                )
                },
                )
adata.var["gene_name"] = adata.var.index.tolist()


# %% [markdown]
# ### 2.4 Tokenize the input data for model fine-tuning
all_counts = (
    adata.layers[config.input_layer_key].A
    if issparse(adata.layers[config.input_layer_key])
    else adata.layers[config.input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)
# %%
if config.use_mod:
    mod_type = np.array([gene_loc_df.loc[g, 'mod'] for g in genes])
    vocab_mod = Vocab(VocabPybind(np.unique(gene_loc_df['mod']).tolist() + special_tokens, None))
    vocab_mod.set_default_index(vocab_mod["<pad>"])
    mod_type = np.array(vocab_mod(list(mod_type)), dtype=int)
    ntokens_mod = len(vocab_mod)

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

# %%
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
num_of_non_zero_genes = [
    np.count_nonzero(train_data[i]) for i in range(train_data.shape[0])
]
logger.info(f"max num of non_zero genes: {np.max(num_of_non_zero_genes)}")
logger.info(f"min num of non_zero genes: {np.min(num_of_non_zero_genes)}")
logger.info(f"average num of non_zero genes: {np.mean(num_of_non_zero_genes)}")
logger.info(f"99% quantile num of non_zero genes: {np.quantile(num_of_non_zero_genes, 0.99)}")
logger.info(f"max original values: {np.max(train_data)}")
logger.info(f"average original non_zero values: {np.mean(train_data[np.nonzero(train_data)])}")
logger.info(f"99% quantile original non_zero values: {np.quantile(train_data[np.nonzero(train_data)], 0.99)}")
logger.info(f"num of celltypes: {num_types}")

# %%
if config.load_model is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
else:
    pretrained_genes = [g for g in genes + special_tokens if g in old_vocab]
    new_genes = [g for g in genes + special_tokens if g not in old_vocab]
    gene_ids_pretrained = np.array(old_vocab(pretrained_genes), dtype=int)
    # https://discuss.pytorch.org/t/expand-an-existing-embedding-and-linear-layer-nan-loss-value/55670/2
    # Retrieve pretrained weights
    vocab = Vocab(VocabPybind(pretrained_genes + new_genes, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
# %%
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=config.include_zero_gene,
    mod_type=mod_type if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
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
    mod_type=mod_type if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
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
# ## Step 3: Load the pre-trained scGPT model
# Note that for multiomic integration, since the pre-trained model does not include the ATAC and protein tokens, we expand the embedding layer by adding these new tokens. We inherit only the gene embedding layer from the pre-trained model, and train rest of the model from scratch.


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict = torch.load(model_file)
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
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=config.DAR,
    use_batch_labels=config.use_batch_labels,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    pre_norm=config.pre_norm,
    use_mod=config.use_mod,
    ntokens_mod=ntokens_mod if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
)

with torch.no_grad():
    pretrained_emb_weights = model_dict['encoder.embedding.weight'][gene_ids_pretrained, :]
    model.encoder.embedding.weight.data[:len(pretrained_genes), :] = pretrained_emb_weights
    model.encoder.enc_norm.weight.data = model_dict['encoder.enc_norm.weight']
ntokens = len(vocab)
model.to(device)
logger.info(model)

# %%
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.schedule_interval, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# %% [markdown]
# ## Step 4: Finetune scGPT with task-specific objectives

# %%
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
        shuffle=True,
        intra_domain_shuffle=False,
        drop_last=False,
        per_seq_batch_sample=config.per_seq_batch_sample
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
        per_seq_batch_sample=config.per_seq_batch_sample
    )

    if config.do_train:
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
        epoch=epoch
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss {val_loss:5.4f} | "
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
            model = best_model,
            adata_t = adata,
            gene_ids = gene_ids,
            vocab = vocab,
            config = config,
            logger = logger,
            include_types=["cls"],
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

gc.collect()