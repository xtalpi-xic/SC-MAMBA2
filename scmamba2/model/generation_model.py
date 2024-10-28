import os
import math
from typing import Mapping, Optional, Tuple, Any, Union

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.utils.data import dataset
from tqdm import trange

from .model import (
    GeneEncoder,
    ExprDecoder,
    ContinuousValueEncoder,
    mambaformer,
)
from ..utils import map_raw_id_to_vocab_id
from .. import logger


class MambaGenerator(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        mamba_layer: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        pert_pad_id: int = 2,
        domain_spec_batchnorm: Union[bool, str] = False,
        n_input_bins: Optional[int] = 0,
        cell_emb_style: str = "cls",
        decoder_activation: Optional[str] = None,
        decoder_adaptive_bias: bool = False,
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.model_type = "Mamba"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.ecs_threshold = ecs_threshold
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        # TODO: the way the pert flag is added is a bit tricky, may update to only add to the perturbed genes

        # print("Using simple batchnorm instead of domain specific batchnorm")
        # self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)  # TODO: test whether do batchnorm
        self.transformer_encoder = mambaformer(d_model, mamba_layer)

        # self.decoder = nn.Linear(d_model, 1)
        self.decoder = AffineExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            activation=decoder_activation,
            adaptive_bias=decoder_adaptive_bias,
        )
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        total_embs = src + values + perts

        # total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.mamba_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)


    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
        Returns:
            dict of output Tensors.
        """
        if self.explicit_zero_prob and not do_sample and not self.training:
            do_sample = True
            logger.warning("Auto set do_sample to True when model is in eval mode.")

        # binning input gene values
        if self.n_input_bins > 0:
            from ..preprocess import binning

            processed_values = torch.stack(
                [binning(row, n_bins=self.n_input_bins) for row in values], dim=0
            ).to(values.device)
        else:
            processed_values = values

        mamba_output = self._encode(
            src, processed_values, input_pert_flags, src_key_padding_mask
        )
        output = {}
        
        mlm_output = self.decoder(mamba_output, values)  
        
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        output_to_cpu: bool = True,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len]
            values: Tensor, shape [N, seq_len]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        outputs = []
        N = src.size(0)
        device = next(self.parameters()).device
        for i in trange(0, N, batch_size):
            output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
            )
            if output_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, -1)

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,  # TODO: Check if this is correct? If not trained the zero probs in training, how come use that in evaluation?
                )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class AffineExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        activation: Optional[str] = None,
        tanh_coeff: bool = False,
        adaptive_bias: bool = False,
    ):
        """
        Predict the expression value of each gene in an affine like form of Ax + b.
        This decoder takes two ExprDecoder intrinsically to genrate the coefficient A and bias b.

        Args:
            d_model: The embedding dimension.
            explicit_zero_prob: If True, predict the probability of each gene being
                zero.
            activation: The activation function for the coefficient A and bias b.
            tanh_coeff: If True, use tanh activation for the coefficient A.
            adaptive_bias: If True, use a learnable bias for the bias b.
        """
        super().__init__()
        self.explicit_zero_prob = explicit_zero_prob
        self.tanh_coeff = tanh_coeff
        self.adaptive_bias = adaptive_bias
        self.coeff_decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)
        self.bias_decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)

        self.activation = activation
        if activation is not None:
            assert hasattr(nn, activation), f"Unknown activation: {activation}"
            self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor, values: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embsize]
            values: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len]
        """
        coeff = self.coeff_decoder(x)
        bias = self.bias_decoder(x)
        # TODO: the bias can try relu activation as well

        if self.activation is not None:
            coeff["pred"] = self.activation(coeff["pred"])
            bias["pred"] = self.activation(bias["pred"])

        # if self.tanh_coeff:
        #     coeff["pred"] = 1 + torch.tanh(coeff["pred"])

        if self.adaptive_bias:
            # bias["pred"] = bias["pred"] * values.mean(dim=1, keepdim=True)
            non_zero_value_mean = values.sum(dim=1, keepdim=True) / (values != 0).sum(
                dim=1, keepdim=True
            )
            bias["pred"] = bias["pred"] * non_zero_value_mean

        if self.explicit_zero_prob:
            return {
                "pred": coeff["pred"] * values + bias["pred"],
                "zero_probs": coeff["zero_probs"],
            }

        return dict(pred=coeff["pred"] * values + bias["pred"])



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



