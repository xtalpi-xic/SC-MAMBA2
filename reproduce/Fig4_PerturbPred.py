# %%
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import torch
from gears import PertData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from torch import nn
from torch_geometric.loader import DataLoader
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab
from types import SimpleNamespace

from scmamba.model import MambaGenerator

import scmamba
from scmamba.loss import masked_mse_loss, masked_relative_error
from scmamba.tokenizer.gene_tokenizer import GeneVocab
from scmamba.utils import map_raw_id_to_vocab_id, set_seed, compute_perturbation_metrics


matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

# %% [markdown]
#  ## Training Settings

# %%

config = SimpleNamespace(
    seed=0,
    data_name="norman", #"norman" #"adamson" #"k562_1900_100_re_ctrl_sample"
    # load_model="/home/bowen.zhao/work/scMamba_fine_tuning/save/2024_whole_body/",
    load_model='/mnt/nas/data/personal/fan.zhang/2_Transcript/scMamba_exp/model_save/exp20-SC-MAMBA-24-BI-Smart-Flip-Share-Wholebody-2023-epoch6',
    #"../save/scGPT_human",
    mapping_23_to_24 = False,
    max_seq_len=6000,
    lr=1e-4,
    batch_size=16,
    layer_size=512,
    nlayers=12,
    nhead=8,
    n_bins=0,
    dropout=0,
    epochs=1,
    early_stop=10,
    decoder_activation=None,
    decoder_adaptive_bias=False,
    log_interval=100,
    amp=True,
)

set_seed(config.seed)

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # TODO: add loading the value from model config
pert_pad_id = 0  # TODO: the value is for adding pad pert tokens, may not nused if all inputs are the same length. The value should selected aside of the actual pert tokens used in the data.
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False

# settings for training
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
eval_batch_size = config.batch_size
schedule_interval = 1

# settings for the model
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in transformer
nlayers = config.nlayers  # number of transformer layers
nhead = config.nhead  # number of heads in nn.MultiheadAttention
n_layers_cls = 3


# dataset and evaluation choices
data_name: str = config.data_name
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]
elif data_name.startswith("k562_1900_"):
    perts_to_plot = ["GATA1+ctrl"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
save_dir = Path(f"./save/dev_perturb_mamba_2024wholebody_6epoch_1gpu_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

logger = scmamba.logger
scmamba.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


# %%
if data_name.startswith("k562_1900_"):
    data_dir = "/scratch/ssd004/scratch/chloexq/perturb_analysis/{}".format(data_name)
    pert_data = PertData(data_dir)
    pert_data.load(data_path=data_dir)
    dataset_processed = pert_data.dataset_processed
    
    #### map 2023 gene name to 2024
    if config.mapping_23_to_24:
        logger.info('Map gene name from 2023 to 2024')
        pert_data = PertData("./data")
        pert_data.load(data_name=data_name)
        df = pd.read_csv('gene_name_2024.csv',sep='\t')
        pert_data.adata.var.reset_index(inplace=True)
        pert_data.adata.var = pert_data.adata.var.merge(df, left_on='gene_id', right_on='feature_id', how='left') \
            .assign(gene_name=lambda x: x['feature_name'].where(x['feature_name'].notna(), x['gene_name'])) \
            .drop(['feature_id','feature_name'], axis=1) \
            .set_index('gene_id')
else:
    pert_data = PertData("./data")
    pert_data.load(data_name=data_name)

# %%
pert_data.prepare_split(split=split, seed=1)
if "ctrl" in pert_data.set2conditions["train"]:
    pert_data.set2conditions["train"].remove("ctrl")
    logger.info("Remove ctrl condition from training set")
if "ctrl" in pert_data.set2conditions["val"]:
    pert_data.set2conditions["val"].remove("ctrl")
    logger.info("Remove ctrl condition from validation set")
pert_data.get_dataloader(batch_size=config.batch_size, test_batch_size=eval_batch_size)


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

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
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
)
n_genes = len(genes)

# %% [markdown]
#  # Create and train scGpt

# %%
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
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    n_input_bins=config.n_bins,
    decoder_activation=config.decoder_activation,
    decoder_adaptive_bias=config.decoder_adaptive_bias,
)
if load_param_prefixs is not None and config.load_model is not None:
    # only load params that start with the prefix
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
elif config.load_model is not None:
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
wandb.watch(model)


# %%

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        # TODO: try add the size factor to the model decoder as well
        # TODO: append cls
        # TODO: predict Ax + b style as the output predictions
        # TODO: Is there a way to input only non-zero genes



        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > config.max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    : config.max_seq_len
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

        with torch.cuda.amp.autocast(enabled=config.amp, dtype=torch.bfloat16):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
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

        wandb.log({"train/loss": loss.item()})

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.4f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


# %%
def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float)

    return results


# %%
best_val_loss = float("inf")
best_val_corr = 0
best_model = None
patience = 0

train_loader = pert_data.dataloader["train_loader"]
valid_loader = pert_data.dataloader["val_loader"]
for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()

    train(model, train_loader,)

    val_res = eval_perturb(valid_loader, model, device)
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    logger.info(f"val_metrics at epoch {epoch}: ")
    logger.info(val_metrics)
    wandb.log({f"valid/{k}": v for k, v in val_metrics.items()})

    # test_loader = pert_data.dataloader["test_loader"]
    # test_res = eval_perturb(test_loader, model, device)
    # test_metrics = compute_perturbation_metrics(
    #     test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    # )
    # logger.info(f"test_metrics at epoch {epoch}: ")
    # logger.info(test_metrics)
    # wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    # torch.save(model.state_dict(), save_dir / f"model_{epoch}.pt")

    elapsed = time.time() - epoch_start_time
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    val_score = val_metrics["pearson"]
    if val_score > best_val_corr:
        best_val_corr = val_score
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {val_score:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= config.early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    # torch.save(
    #     model.state_dict(),
    #     save_dir / f"model_{epoch}.pt",
    # )

    scheduler.step()  # TODO: have this back


# %%
torch.save(best_model.state_dict(), save_dir / "best_model.pt")


# %% [markdown]
#  ## Evaluations


# %%
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
                    batch_data,
                    include_zero_gene=include_zero_gene,
                    gene_ids=gene_ids,
                    amp=config.amp,
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred


# %%
def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None
) -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

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

    fig, ax = plt.subplots(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        fig.savefig(save_file, bbox_inches="tight", transparent=False)

    return fig


# %%
# predict(best_model, [["FEV"], ["FEV", "SAMD11"]])
for p in perts_to_plot:
    fig = plot_perturbation(
        best_model, p, pool_size=300, save_file=f"{save_dir}/{p}.png"
    )
    wandb.log({p: wandb.Image(fig)})


# %%
test_loader = pert_data.dataloader["test_loader"]
test_res = eval_perturb(test_loader, best_model, device)
# test_metrics, test_pert_res = compute_metrics(test_res)
test_metrics = compute_perturbation_metrics(
    test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
)
print(test_metrics)

# save the dicts in json
with open(f"{save_dir}/test_metrics.json", "w") as f:
    json.dump(test_metrics, f)
# with open(f"{save_dir}/test_pert_res.json", "w") as f:
#     json.dump(test_pert_res, f)

wandb.log(test_metrics)

deeper_res = deeper_analysis(pert_data.adata, test_res)
non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

metrics = ["pearson_delta", "pearson_delta_de"]
metrics_non_dropout = [
    "pearson_delta_top20_de_non_dropout",
    "pearson_top20_de_non_dropout",
    'frac_opposite_direction_top20_non_dropout',
    'frac_sigma_below_1_non_dropout',
    'mse_top20_de_non_dropout'
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
        mean_value = np.mean(subgroup_analysis[name][m])
        logger.info("test_" + name + "_" + m + ": " + str(mean_value))
        if not np.isnan(mean_value):
            wandb.log({f"test/{name}_{m}": mean_value})

# %%
wandb.finish()