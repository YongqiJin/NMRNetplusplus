##################################################################################################
# Minimal-diff version of unimat_solv with four solvent modes controlled by two new flags:
#  - --bos-only (only effective when --solvent-embed-before-backbone True)
#  - --solv-concat (only effective when --solvent-embed-after-backbone True)
# Behavior:
#  Before-backbone:
#     bos_only=True  -> add only to BOS token
#     bos_only=False -> broadcast add to all tokens (original behavior)
#  After-backbone:
#     solv_concat=True  -> original concat + linear projection
#     solv_concat=False -> direct additive broadcast (no extra linear)


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

@register_model("unimat_solv")
class UniMatModelwithSolvent(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        # Copy original arguments
        parser.add_argument("--solvent-embed-dim", type=int, metavar="H", help="solvent embedding dimension")
        parser.add_argument("--solvent-embed-before-backbone", action="store_true", help="use solvent embedding before backbone")
        parser.add_argument("--solvent-embed-after-backbone", action="store_true", help="use solvent embedding after backbone")
        parser.add_argument("--solvent-max-types", type=int, metavar="F", help="maximum number of solvent types")
        # New flags
        parser.add_argument("--bos-only", action="store_true", help="only add solvent embedding to BOS token when before-backbone mode")
        parser.add_argument("--embed-after-head", action="store_true", help="after-backbone: add scalar to head output (else additive to backbone output)")
        # Original model hyper-parameters
        parser.add_argument("--encoder-layers", type=int, metavar="L", help="num encoder layers")
        parser.add_argument("--encoder-embed-dim", type=int, metavar="H", help="encoder embedding dim")
        parser.add_argument("--encoder-ffn-embed-dim", type=int, metavar="F", help="ffn dim")
        parser.add_argument("--encoder-attention-heads", type=int, metavar="A", help="num attention heads")
        parser.add_argument("--activation-fn", choices=utils.get_available_activation_fns(), help="activation function")
        parser.add_argument("--pooler-activation-fn", choices=utils.get_available_activation_fns(), help="pooler activation fn")
        parser.add_argument("--emb-dropout", type=float, metavar="D", help="embedding dropout")
        parser.add_argument("--dropout", type=float, metavar="D", help="dropout probability")
        parser.add_argument("--attention-dropout", type=float, metavar="D", help="attention dropout")
        parser.add_argument("--activation-dropout", type=float, metavar="D", help="activation dropout")
        parser.add_argument("--pooler-dropout", type=float, metavar="D", help="pooler dropout")
        parser.add_argument("--max-seq-len", type=int, help="max sequence length")
        parser.add_argument("--post-ln", action="store_true", help="use post layernorm")
        parser.add_argument("--masked-token-loss", type=float, metavar="D", help="mask token loss ratio")
        parser.add_argument("--masked-dist-loss", type=float, metavar="D", help="masked distance loss ratio")
        parser.add_argument("--masked-coord-loss", type=float, metavar="D", help="masked coord loss ratio")
        parser.add_argument("--x-norm-loss", type=float, metavar="D", help="x norm loss ratio")
        parser.add_argument("--delta-pair-repr-norm-loss", type=float, metavar="D", help="delta pair repr norm loss ratio")
        parser.add_argument("--masked-coord-dist-loss", type=float, metavar="D", help="masked coord dist loss ratio")
        parser.add_argument("--lattice-loss", type=float, metavar="D", help="lattice loss ratio")
        parser.add_argument("--gaussian-kernel", action="store_true", help="use gaussian kernel")
        # parser.add_argument("--atom-descriptor", action="store_true", help="use extra atom descriptor")
        # parser.add_argument("--global-distance", action="store_true", help="use global distance")

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), args.encoder_embed_dim, self.padding_idx)

        if args.solvent_embed_before_backbone:
            self.solvent_embed_full = nn.Embedding(args.solvent_max_types, args.encoder_embed_dim)

        # After-backbone embeddings:
        # If embed_after_head is True, we use a scalar embedding added to the logits.
        # If embed_after_head is False, we use additive injection to the backbone output (Scenario D).
        if args.solvent_embed_after_backbone:
            if args.embed_after_head:
                self.solvent_embed_scalar = nn.Embedding(args.solvent_max_types, 1)
                self.solvent_embed_after_add = None
            else:
                self.solvent_embed_scalar = None
                self.solvent_embed_after_add = nn.Embedding(args.solvent_max_types, args.encoder_embed_dim)
        else:
            self.solvent_embed_scalar = None
            self.solvent_embed_after_add = None

        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(embed_dim=args.encoder_embed_dim,
                                      output_dim=len(dictionary),
                                      activation_fn=args.activation_fn,
                                      weight=self.embed_tokens.weight)

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(K, args.encoder_attention_heads, args.activation_fn)
        if 'gaussian_kernel' in args and args.gaussian_kernel:
            if args.global_distance:
                self.gbf = GlobalGaussianLayer(K, n_edge_type)
            else:
                self.gbf = GaussianLayer(K, n_edge_type)
        else:
            self.gbf = NumericalEmbed(K, n_edge_type)
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(args.encoder_attention_heads, 1, args.activation_fn)
        if args.masked_dist_loss > 0:
            if args.global_distance:
                self.glodist_head = GlobalDistanceHead(args.encoder_attention_heads, args.activation_fn)
            else:
                self.dist_head = DistanceHead(args.encoder_attention_heads, args.activation_fn)
        if args.lattice_loss > 0:
            self.lattice_head = NodeClassificationHead(input_dim=args.encoder_embed_dim,
                                                       inner_dim=args.encoder_embed_dim,
                                                       num_classes=6,
                                                       activation_fn=self.args.pooler_activation_fn,
                                                       pooler_dropout=self.args.pooler_dropout)
        self.classification_heads = nn.ModuleDict()
        self.node_classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary)

    def forward(self,
        select_atom,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        solvent,
        atom_descriptor=None,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs):

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = et.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)

        # BEFORE BACKBONE INJECTION (broadcast or BOS-only)
        if self.args.solvent_embed_before_backbone and solvent is not None:
            solvent_embeds = self.solvent_embed_full(solvent.long())  # (B, D)
            if self.args.bos_only:
                # Out-of-place modification to avoid in-place gradient side-effects
                x = x.clone()
                x[:, 0, :] = x[:, 0, :] + solvent_embeds
            else:
                solvent_embeds = solvent_embeds.unsqueeze(1).expand(-1, x.size(1), -1)
                x = x + solvent_embeds

        encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm = self.encoder(
            x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float('-inf')] = 0

        encoder_distance = None
        encoder_coord = None
        lattice = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(-1, 1, 1, 1)
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                if self.args.global_distance:
                    encoder_distance = self.glodist_head(encoder_pair_rep)
                else:
                    encoder_distance = self.dist_head(encoder_pair_rep)
            if self.args.lattice_loss > 0:
                lattice = self.lattice_head(encoder_rep[select_atom == 1])
        if atom_descriptor is not None:
            model_input = torch.cat((encoder_rep, atom_descriptor.to(encoder_rep.dtype)), dim=2)
        else:
            model_input = encoder_rep

        # AFTER BACKBONE INJECTION
        #  规则：
        #   - embed_after_head=True: 不修改 model_input，在 logits 处加 scalar
        #   - embed_after_head=False: 使用 encoder_embed_dim 维 embedding (self.solvent_embed_after_add) 逐 token 相加
        if self.args.solvent_embed_after_backbone and solvent is not None:
            if not self.args.embed_after_head and self.solvent_embed_after_add is not None:
                # additive after-backbone: 使用 full-dim embedding (= encoder_embed_dim)
                solvent_embeds2 = self.solvent_embed_after_add(solvent.long())  # (B, D)
                solvent_exp = solvent_embeds2.unsqueeze(1).expand(-1, model_input.size(1), -1)
                model_input = model_input + solvent_exp

        if classification_head_name is not None:
            if classification_head_name in self.classification_heads:
                logits = self.classification_heads[classification_head_name](model_input)
            elif classification_head_name in self.node_classification_heads:
                logits = self.node_classification_heads[classification_head_name](model_input[select_atom == 1])
        elif features_only and (self.classification_heads or self.node_classification_heads):
            logits = {}
            for name, head in self.node_classification_heads.items():
                logits[name] = head(model_input[select_atom == 1])
            for name, head in self.classification_heads.items():
                logits[name] = head(model_input)

        # Apply scalar solvent embedding if embed_after_head is True
        if self.args.solvent_embed_after_backbone and self.args.embed_after_head and solvent is not None:
            solvent_scalar = self.solvent_embed_scalar(solvent.long()) # (B, 1)
            
            def apply_solvent_scalar(output, is_node_level):
                if is_node_level:
                    # output: (Total_Atoms, num_classes)
                    # solvent_scalar: (B, 1)
                    # select_atom: (B, S)
                    # Expand solvent to (B, S, 1)
                    sol_exp = solvent_scalar.unsqueeze(1).expand(-1, select_atom.size(1), -1)
                    # Flatten to (Total_Atoms, 1)
                    sol_flat = sol_exp[select_atom == 1]
                    return output + sol_flat
                else:
                    # output: (B, num_classes)
                    return output + solvent_scalar

            if isinstance(logits, torch.Tensor):
                if classification_head_name in self.node_classification_heads:
                    logits = apply_solvent_scalar(logits, True)
                elif classification_head_name in self.classification_heads:
                    logits = apply_solvent_scalar(logits, False)
            elif isinstance(logits, dict):
                for name in logits:
                    if name in self.node_classification_heads:
                        logits[name] = apply_solvent_scalar(logits[name], True)
                    elif name in self.classification_heads:
                        logits[name] = apply_solvent_scalar(logits[name], False)

        return logits, encoder_distance, encoder_coord, lattice, x_norm, delta_encoder_pair_rep_norm

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, extra_dim=0, **kwargs
    ):
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        input_dim = self.args.encoder_embed_dim + extra_dim
        self.classification_heads[name] = ClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim or input_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def register_node_classification_head(
        self, name, num_classes=None, inner_dim=None, extra_dim=0, **kwargs
    ):
        if name in self.node_classification_heads:
            prev_num_classes = self.node_classification_heads[name].out_proj.out_features
            prev_inner_dim = self.node_classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        input_dim = self.args.encoder_embed_dim + extra_dim
        self.node_classification_heads[name] = NodeClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim or input_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))
    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NodeClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class DistanceHead(nn.Module):
    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)
    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x

class GlobalDistanceHead(nn.Module):
    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 4)
        self.activation_fn = utils.get_activation_fn(activation_fn)
    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        x = (x + x.transpose(1, 2)) * 0.5
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class GlobalGaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x + bias
        replication_rules = [128, 0, 0, 0]
        x = torch.repeat_interleave(x, torch.tensor(replication_rules).to(x.device), dim=-1)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NumericalEmbed(nn.Module):
    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        super().__init__()
        self.K = K
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.w_edge = nn.Embedding(edge_types, K)
        self.proj = NonLinearHead(1, K, activation_fn, hidden=2*K)
        self.ln = LayerNorm(K)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.kaiming_normal_(self.w_edge.weight)
    def forward(self, x, edge_type):
        # Support both shapes: (B, N, N, 1) and (B, N, N)
        if x.dim() == 4:
            x = x[:, :, :, 0]
        elif x.dim() == 3:
            # already (B, N, N)
            pass
        else:
            raise RuntimeError(f"NumericalEmbed expects x dim 3/4, got {x.shape}")
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        w_edge = self.w_edge(edge_type).type_as(x)
        edge_emb = w_edge * torch.sigmoid(mul * x.unsqueeze(-1) + bias)
        edge_proj = x.unsqueeze(-1).type_as(self.mul.weight)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)
        h = edge_proj + edge_emb
        h = h.type_as(self.mul.weight)
        return h

@register_model_architecture("unimat_solv", "unimol_large_solv")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.lattice_loss = getattr(args, "lattice_loss", -1.0)
    args.solvent_max_types = getattr(args, "solvent_max_types", 4)
    args.solvent_embed_dim = getattr(args, "solvent_embed_dim", 0)
    args.solvent_embed_after_backbone = getattr(args, "solvent_embed_after_backbone", False)
    args.solvent_embed_before_backbone = getattr(args, "solvent_embed_before_backbone", False)
    args.bos_only = getattr(args, "bos_only", False)
    args.embed_after_head = getattr(args, "embed_after_head", False)
    args.post_ln = getattr(args, "post_ln", False)
