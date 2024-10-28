# WANDB runs
# https://wandb.ai/scformer-team/scGPT-public/runs/pdcehvsz?workspace=user-
# https://wandb.ai/scformer-team/scGPT-public/runs/xrrstivq?workspace=user-
# https://wandb.ai/scformer-team/scGPT-public/runs/2du27x2b?workspace=user-
# https://wandb.ai/scformer-team/scGPT-public/runs/y0bu7wuz?workspace=user-
# https://wandb.ai/scformer-team/scGPT-public/runs/hrwfsxzw?workspace=user-
# In[1]:
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")
from scgpt.model import TransformerGenerator

import scgpt as scg
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

import wandb

hyperparameter_defaults = dict(
    seed=0,
    load_transformer = True,
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="reverse_pert",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)

config = wandb.config

set_seed(config.seed)

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
max_seq_len = 1536
n_bins = 0

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True
# load_model = '/home/bowen.zhao/work/scMamba_fine_tuning/save/6_epoch'
load_model = '/mnt/nas/data/personal/fan.zhang/2_Transcript/scMamba_exp/model_save/exp20-SC-MAMBA-24-BI-Smart-Flip-Share-Wholebody-2023-epoch6'
load_param_prefixs = [
        "encoder",
        "value_encoder"]
if config.load_transformer:
    load_param_prefixs.append("transformer_encoder")
print(load_param_prefixs)

# settings for optimizer
lr = 1e-4
batch_size = 64
eval_batch_size = 64
epochs = 15
schedule_interval = 1
schedule_ratio=0.5
early_stop = 4

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.3  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

decoder_activation=None
decoder_adaptive_bias=False
# logging
log_interval = 100

# dataset and evaluation choices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


data_name = 'norman_sub'
save_dir = Path(f"./save/dev_reverse_perturb_2023wholebody_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

data_dir = "/home/fan.zhang/2_Transcript/scGPT_processed_datasets/Fig3_Perturbation/Fig3_FG_ReversePerturb/norman_ctrl_500"
# Note that the norman_ctrl_500 data subset contains the Train and Valid perturbations only
# For testing, we load the original norman dataset again in later steps
pert_data = PertData(data_dir)
pert_data.load(data_path=data_dir)
#pert_data.prepare_split(split='simulation', seed=1)
#pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
dataset_processed = pert_data.dataset_processed
print(len(dataset_processed['ctrl']))
#index = np.random.choice(len(dataset_processed['ctrl']), 500, replace=False)
#dataset_processed['ctrl'] = [dataset_processed['ctrl'][i] for i in index]
pert_data.prepare_split(split = 'no_test', seed = 1)
# TODO
pert_data.set2conditions['train'].remove('ctrl')
pert_data.get_dataloader(batch_size=batch_size)
count = 0
for i, data in enumerate(iter(pert_data.dataloader['train_loader'])):
    count+=sum([p=='ctrl' for p in data.pert])
print(count)


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


#  # Create and train scGpt

# In[6]:


ntokens = len(vocab)  # size of vocabulary
model = TransformerGenerator(
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
    decoder_activation=decoder_activation,
    decoder_adaptive_bias=decoder_adaptive_bias,
    use_fast_transformer=use_fast_transformer,
)

if load_param_prefixs is not None and load_model is not None:
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
elif load_model is not None:
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



# In[7]:


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=schedule_ratio)
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

        # torch.cuda.empty_cache()

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
                f"lr {lr:05.6f} | ms/batch {ms_per_batch:5.2f} | "
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
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
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
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp,dtype=torch.bfloat16):
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
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)



# In[8]:


train_loader = pert_data.dataloader["train_loader"]


# In[9]:


x = next(iter(train_loader))


# In[10]:


pert_data.adata[(pert_data.adata.obs['split']=='train') & (pert_data.adata.obs['condition']=='ctrl'), :]
#not control: 19324
#control: 7353

# In[12]:


# CNN1+MAPK1
# FOSB+IKZF3
# FOSB+UBASH3B
# FOXA1+HOXB9
# IGDCC3+MAPK1
# MAPK1+PRTG
# PTPN9+UBASH3B


# In[ ]:


best_val_loss = float("inf")
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    train(
        model,
        train_loader,
    )
    val_loss, val_mre = evaluate(
        model,
        valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.6f} |"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {best_val_loss:5.6f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    # torch.save(
    #     model.state_dict(),
    #     save_dir / f"model_{epoch}.pt",
    # )

    scheduler.step()


# In[ ]:
torch.save(best_model.state_dict(), save_dir / "best_model.pt")


#  ## Evaluations

# In[56]:

model = best_model


data_dir = Path('/home/fan.zhang/2_Transcript/scGPT_processed_datasets/Fig3_Perturbation/Fig3_FG_ReversePerturb/')
data_name = "norman"
split = "simulation"
pert_data_ = PertData(data_dir)
pert_data_.load(data_name=data_name)
pert_data_.prepare_split(split=split, seed=1)

pert_data_.adata


# In[82]:


import numpy as np
try:
    np.unique(pert_data_.adata.obs['split'].values)
except:
    pert_data_.adata.obs['split'] = ''
    pert_data_.adata.obs.loc[pert_data_.adata.obs['condition'].isin(pert_data_.set2conditions["train"]), 'split'] = 'train'
    pert_data_.adata.obs.loc[pert_data_.adata.obs['condition'].isin(pert_data_.set2conditions["val"]), 'split'] = 'val' #'test' 
    pert_data_.adata.obs.loc[pert_data_.adata.obs['condition'].isin(pert_data_.set2conditions["test"]), 'split'] = 'test' #'ood' #'test'
    assert len(np.unique(pert_data_.adata.obs['split'].values)) == 3


# In[83]:


pert_data_.adata.obs['split']


# In[84]:


genes = pert_data_.adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)


# In[85]:


test_groups = pert_data_.subgroup["test_subgroup"].copy()


# In[86]:


test_gene_list = []
for i in test_groups.keys():
    for g in test_groups[i]:
        if g.split('+')[0] != 'ctrl':
            test_gene_list.append(g.split('+')[0])
        if g.split('+')[1] != 'ctrl':
            test_gene_list.append(g.split('+')[1])
test_gene_list = list(set(test_gene_list))


# In[87]:


len(test_gene_list)


# In[88]:


import pandas as pd


# In[89]:


df = pd.DataFrame(np.zeros((len(test_gene_list), len(test_gene_list))), columns = test_gene_list, index = test_gene_list)


# In[90]:


train_condition_list = pert_data_.adata.obs[pert_data_.adata.obs.split=='train'].condition.values
valid_condition_list = pert_data_.adata.obs[pert_data_.adata.obs.split=='val'].condition.values
test_condition_list = pert_data_.adata.obs[pert_data_.adata.obs.split=='test'].condition.values


# In[91]:


def update_df(df, condition_list, label):
    for i in condition_list:
        if i != 'ctrl':
            g0 = i.split('+')[0]
            g1 = i.split('+')[1]
            if g0 == 'ctrl' and g1 in test_gene_list:
                df.loc[g1, g1] = label
            elif g1 == 'ctrl' and g1 in test_gene_list:
                df.loc[g0, g0] = label
            elif g0 in test_gene_list and g1 in test_gene_list:
                df.loc[g0, g1] = label
                df.loc[g1, g0] = label


# In[92]:


update_df(df, train_condition_list, 'Train')
update_df(df, valid_condition_list, 'Valid')
update_df(df, test_condition_list, 'Test')


# In[93]:


df = df.replace({0:'Unseen'})


# In[94]:


sub_gene_list = list(set(df[(df=='Train').sum(0)>0].index).intersection(df[(df=='Test').sum(0)>0].index))
sub_test_gene_list = ((df.loc[:, sub_gene_list]=='Train').sum(0)+(df.loc[:, sub_gene_list]=='Test').sum(0)).sort_values()[-20:].index
sub_df = df.loc[sub_test_gene_list, sub_test_gene_list]
#df = df.loc[np.sort(sub_df.index)[::-1], np.sort(sub_df.index)[::-1]]
df = df.loc[np.sort(sub_df.index), np.sort(sub_df.index)]


# In[95]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# In[96]:


plt.figure(figsize=(11, 10))
value_to_int = {j:i for i,j in enumerate(['Unseen', 'Train', 'Valid', 'Test'])}
n = len(value_to_int)   
cmap = sns.color_palette("light:slateblue", as_cmap=True)
matrix = np.triu(df.values, 1)
ax = sns.heatmap(df.replace(value_to_int), cmap=mpl.colors.ListedColormap(cmap(np.linspace(0, 1, 4))), linewidths=0.05, mask=matrix) 
ax.tick_params(axis='y', rotation=0)
ax.tick_params(axis='x', rotation=90)
colorbar = ax.collections[0].colorbar 
r = colorbar.vmax - colorbar.vmin 
colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys())) 
plt.show()


# In[97]:


test_gene_list = df.index.tolist()


# In[98]:


train_num = (df.mask(~np.triu(np.ones(df.shape, dtype=np.bool_)))=='Train').sum().sum()
valid_num = (df.mask(~np.triu(np.ones(df.shape, dtype=np.bool_)))=='Valid').sum().sum()
test_num = (df.mask(~np.triu(np.ones(df.shape, dtype=np.bool_)))=='Test').sum().sum()
total_num = df.shape[0]**2-(df.mask(~np.triu(np.ones(df.shape, dtype=np.bool_))).isnull()).sum().sum()
print('{}/{} train conditions, {}/{} valid conditions, and {}/{} test conditions.'.format(train_num, total_num, valid_num, total_num, test_num, total_num))


# In[105]:


from typing import Dict, Mapping, Optional, Tuple, Any, Union, List

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
    #gene_list = pert_data.gene_names.values.tolist()
    gene_list = pert_data.adata.var.gene_name.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                print(i)
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        results_cell_emb = {}
        for pert in tqdm.tqdm(pert_list):
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=batch_size, shuffle=False)
            preds = []
            cell_embs = []
            for batch_data in loader:
#                 pred_gene_values, cell_emb = model.pred_perturb_cell_emb(
#                     batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
#                 )
                pred_gene_values = model.pred_perturb(
                    batch_data,
                    #n_bins=n_bins,
                    include_zero_gene=include_zero_gene,
                    gene_ids=gene_ids
                )
                preds.append(pred_gene_values)
                #cell_embs.append(cell_emb)
            preds = torch.cat(preds, dim=0)
            #cell_embs = torch.cat(cell_embs, dim=0)
            #results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
            #results_cell_emb["_".join(pert)] = np.mean(cell_embs.detach().cpu().numpy(), axis=0)
            results_pred["_".join(pert)] = preds.detach().cpu().numpy()
            #results_cell_emb["_".join(pert)] = cell_embs.detach().cpu().numpy()

    return results_pred #, results_cell_emb


# In[106]:


import itertools
pert_list = []
for comb in itertools.combinations(test_gene_list + ['ctrl'], 2):
    if comb[0] == 'ctrl':
        pert_list.append([comb[1]])
    elif comb[1] == 'ctrl':
        pert_list.append([comb[0]])
    else:
        pert_list.append([comb[0], comb[1]])


# In[107]:


len(pert_list), pert_list


# In[108]:


#from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from torch_geometric.loader import DataLoader
import tqdm


# In[109]:


np.random.seed(config.seed)
#results_pred, results_cell_emb = predict(model, pert_list, pool_size = 30)
results_pred = predict(model, pert_list, pool_size = 30) #40


# In[110]:


results_pred


# In[111]:


len(pert_list)


# In[112]:


results_pred_np = []
results_cell_emb_np = []
for p in results_pred.keys():
    results_pred_np.append(np.expand_dims(results_pred[p], 0))
    #results_cell_emb_np.append(np.expand_dims(results_cell_emb[p], 0))
results_pred_np = np.concatenate(results_pred_np)
#results_cell_emb_np = np.concatenate(results_cell_emb_np)


# In[113]:


M = results_pred_np.shape[-1]
results_pred_np = results_pred_np.reshape(-1, M)
results_pred_np.shape


# In[114]:


import faiss


# In[115]:


xb = results_pred_np
d = xb.shape[1]
#index = faiss.IndexFlatIP(d)
index = faiss.IndexFlatL2(d)   # build the index, d=size of vectors 
# here we assume xb contains a n-by-d numpy matrix of type float32
#faiss.normalize_L2(xb)
index.add(xb) # add


# In[116]:


test_gene_list


# In[117]:


sub_test_condition_list = []
for c in np.unique(test_condition_list):
    g0 = c.split('+')[0]
    g1 = c.split('+')[1]
    if g0 == 'ctrl' and g1 in test_gene_list:
        sub_test_condition_list.append(c)
    elif g1 == 'ctrl' and g0 in test_gene_list:
        sub_test_condition_list.append(c)
    elif g0 in test_gene_list and g1 in test_gene_list:
        sub_test_condition_list.append(c)
print(sub_test_condition_list)


# In[118]:


q_list = []
ground_truth = []
for c in tqdm.tqdm(sub_test_condition_list):
    g0 = c.split('+')[0]
    g1 = c.split('+')[1]
    if g0 == 'ctrl':
        temp = [g1]
        temp1 = [g1]
    elif g1 == 'ctrl':
        temp = [g0]
        temp1 = [g0]
    else:
        temp = [g0, g1]
        temp1 = [g1, g0]
        if temp in pert_list or temp1 in pert_list:
            sub = pert_data_.adata[pert_data_.adata.obs.split=='test']
            sub = sub[sub.obs.condition==c]
            #q_list.append(sub.X.todense().mean(0))
            q_list.append(sub.X.todense())
            if g0<g1:
                ground_truth.extend([c]*sub.X.todense().shape[0])
            else:
                ground_truth.extend(['+'.join([g1, g0])]*sub.X.todense().shape[0])


# In[119]:


xq = np.concatenate(q_list)
xq.shape


# # Start topk experiment

# In[212]:

# Start topk experiment
count_to_log = {}
ratio_to_log = {}
k = 15        # we want 4 similar vectors
#faiss.normalize_L2(xq)
for k in [1, 2, 3, 4, 5, 6, 7, 8]:
    logger('Top {}'.format(k))
    D, I = index.search(xq, k)
    df = pd.DataFrame(I)
    ind_list = []
    condition_list = []
    ind = 0
    for i in results_pred.keys():
        for j in range(results_pred[i].shape[0]):
            ind_list.append(ind)
            condition_list.append(i)
            ind+=1
    index_to_condition = dict(zip(ind_list, condition_list))
    df = df.replace(index_to_condition)
    df['ground_truth'] = ground_truth
    ground_truth_short = []
    aggr_pred = []
    for i in np.unique(ground_truth):
        values = df[df.ground_truth==i].loc[:, list(range(k))].values.flatten()
        unique, counts = np.unique(values, return_counts=True)
        ind = np.argpartition(-counts, kth=k)[:k]
        aggr_pred.append(np.expand_dims(unique[ind], 0))
        ground_truth_short.append(i)
    df_aggr = pd.DataFrame(np.concatenate(aggr_pred))
    df_aggr['ground_truth'] = ground_truth_short
    pred = df_aggr.values[:, :k]
    truth = df_aggr.values[:, -1]
    count = 0
    for i in range(len(truth)):
        g0 = truth[i].split('+')[0]
        g1 = truth[i].split('+')[1]
        truth0 = '_'.join([g0, g1])
        truth1 = '_'.join([g1, g0])
        if truth0 in pred[i, :] or truth1 in pred[i, :]:
            logger('Correct predicted: truth:', truth0, '; pred', pred[i, :])
            count+=1
        else:
            logger('Error predicted: truth:', truth0, '; pred', pred[i, :])
    logger('2/2:', count)
    count_to_log["Top {} 2/2".format(k)] = count
    ratio_to_log[f"Top {k} 2/2"] = count / len(truth)
    count = 0
    for i in range(len(truth)):
        g0 = truth[i].split('+')[0]
        g1 = truth[i].split('+')[1]
        truth0 = '_'.join([g0, g1])
        truth1 = '_'.join([g1, g0])
        found_one = False
        for j in pred[i, :]:
            if not found_one and (g0 in j or g1 in j):
                found_one = True
                count+=1
    logger('1/2:', count)
    count_to_log["Top {} 1/2".format(k)] = count
    ratio_to_log[f"Top {k} 1/2"] = count / len(truth)
    logger('')
    
# with open(f'{save_dir}/ratio_to_log.json', 'w') as f:
#     json.dump(ratio_to_log, f)

# %%
labels = list(ratio_to_log.keys())
values = list(ratio_to_log.values())
plt.figure(figsize=(8, 5))
# Define the colors for the bars
colors = ['lightcoral' if label.startswith('Top ') and label.split()[1].isdigit() and int(label.split()[1]) <= 15 and label.endswith(' 2/2') else 'lightseagreen' for label in labels]

# Plot the barplot with custom colors
plt.bar(labels, values, color=colors)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Metrics Barplot')
plt.savefig(f'{save_dir}/Metrics Barplot.png')
# Display the plot
plt.show()

wandb.log(ratio_to_log)
run.finish()
wandb.finish()

