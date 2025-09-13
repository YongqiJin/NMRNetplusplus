# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from unicore import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from unicore.losses import UnicoreLoss, register_loss


@register_loss("atom_regloss_mse")   
class AtomRegMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        # self.target_scaler = TargetScaler()
        self.target_scaler = task.target_scaler
        self.count = 5

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], 
                           features_only=True, 
                           classification_head_name=self.args.classification_head_name)
        select_atom = sample["net_input"]["select_atom"].view(-1, 1)
        sample_size = (select_atom==1).sum()
        src_token = sample["net_input"]["src_tokens"].view(-1, 1)
        # _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
        # normalizer = Normalization(_mean, _std, _normal_type)
        is_labeled = sample["is_labeled"] == 1
        num_mask = sample["net_input"]["select_atom"].sum(dim=1)
        id_end = num_mask.cumsum(dim=0)
        sample_labeled = {
            "net_input": {k: v[is_labeled] for k, v in sample["net_input"].items()},
            "target": {k: v[is_labeled] for k, v in sample["target"].items()},
            **{k: v[is_labeled] for k, v in sample.items() if k not in ["net_input", "target"]}
        }
        sample_unlabeled = {
            "net_input": {k: v[~is_labeled] for k, v in sample["net_input"].items()},
            "target": {k: v[~is_labeled] for k, v in sample["target"].items()},
            **{k: v[~is_labeled] for k, v in sample.items() if k not in ["net_input", "target"]}
        }
        net_output_labeled = [net_output[0][end-num:end] for num, end in zip(num_mask[is_labeled], id_end[is_labeled])]
        net_output_unlabeled = [net_output[0][end-num:end] for num, end in zip(num_mask[~is_labeled], id_end[~is_labeled])]
        # import pdb
        # pdb.set_trace() # sample: [399, 875, 1189, 163961, 1100, 5108, 5123, 2024, 198228, 3060]
        if len(net_output_labeled):
            loss, residues = self.compute_loss(torch.concat(net_output_labeled), sample_labeled, reduce=reduce)
        else:
            loss, residues = None, None
        if len(net_output_unlabeled):
            unlabeled_loss, weights, unlabeled_losses = self.compute_unlabeled_loss(net_output_unlabeled, sample_unlabeled, reduce=reduce)
        else:
            unlabeled_loss, weights, unlabeled_losses = None, None, None
        
        if loss is None:
            total_loss = self.args.unlabeled_weight * unlabeled_loss
        elif unlabeled_loss is None:
            total_loss = loss
        else:
            total_loss = loss + self.args.unlabeled_weight * unlabeled_loss
        
        # print("loss", loss, "unlabeled_loss", unlabeled_loss, "total_loss", total_loss)

        # if self.count > 0:
        #     self.count -= 1
        #     import logging
        #     logging.info("total_loss: {}, loss: {}, unlabeled_loss: {}".format(total_loss.item(), loss.item() if loss else None, unlabeled_loss.item() if unlabeled_loss else None))
        #     import pdb
        #     pdb.set_trace()  # sample: [ 399,  875, 1189, 1100, 5108, 5123, 2024, 3060]
            # [sum(item).item() for item in sample_labeled['net_input']['select_atom']]: [21, 3, 5, 6, 14, 2, 5, 6]
            # sample_labeled['target']['finetune_target'].sum(dim=1): 
            # tensor([-63.8207, -46.8866, -22.8279, -36.8886, -83.2713, -47.6656, -21.3024, -55.8986], device='cuda:0')
            # [sum(item).item() for item in net_output_labeled]: 
            # [0.62158203125, -0.00815582275390625, 0.459716796875, -0.79150390625, -0.85205078125, -0.344482421875, 0.493408203125, 0.3017578125]
            # [0.62158203125, 0.040283203125, 0.783203125, 0.82373046875, -0.392578125, 0.1688232421875, -0.2578125, 0.188232421875]
        
        if not self.training:
            logging_output = {
                "predict": self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu()).astype('float32'),
                "target": self.target_scaler.inverse_transform((sample["target"]["finetune_target"].view(-1, self.args.num_classes))[select_atom==1].view(-1, self.args.num_classes).data.cpu()).astype('float32'),
                "src_token": src_token,
                "select_atom": select_atom,
                "sample_size": sample_size,
                "matid": sample["matid"],
                "num_task": self.args.num_classes,
                # "encoder_rep": net_output[6],
            }
            total_loss = loss
            # print("predict_output", self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu()))
            # print("target_out", (sample["target"]["finetune_target"].view(-1, self.args.num_classes))[select_atom==1].view(-1, self.args.num_classes).data.cpu())
            # print("predict", self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu()).astype('float32'))
            # print("target", self.target_scaler.inverse_transform((sample["target"]["finetune_target"].view(-1, self.args.num_classes))[select_atom==1].view(-1, self.args.num_classes).data.cpu()).astype('float32'))
        else:
            logging_output = {
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "total_loss": total_loss.data,
                "weights": weights if weights is not None else torch.empty(0, dtype=torch.float32),
            }
        logging_output.update({
            'bsz': sample_size,
            "loss": loss.data if loss is not None else 0,
            "unlabeled_loss": unlabeled_loss.data if unlabeled_loss is not None else None,
            "labeled_residues": residues if residues is not None else torch.empty(0, dtype=torch.float32),
            "unlabeled_losses": unlabeled_losses if unlabeled_losses is not None else torch.empty(0, dtype=torch.float32),
        })
        # print(logging_output)
        # self.count -= 1
        # if self.count < 0:
        #     import pdb; pdb.set_trace()
        # return loss, sample_size, logging_output      
        return total_loss, sample_size, logging_output      

    # def compute_loss(self, net_output, sample, reduce=True):
    #     select_atom = sample["net_input"]["select_atom"].view(-1, 1)
    #     predicts = net_output.view(-1, self.args.num_classes).float()
    #     targets = sample['target']['finetune_target'].view(-1, self.args.num_classes).float()
    #     # normalize_targets = torch.from_numpy(self.target_scaler.transform(targets.cpu())).to(targets.device).float()
    #     loss = F.mse_loss(    # l1_loss mse_loss
    #         predicts,
    #         targets[select_atom==1].view(-1, self.args.num_classes),
    #         # reduction="sum" if reduce else "none",
    #     )
    #     residues = predicts - targets[select_atom==1].view(-1, self.args.num_classes)
    #     return loss, residues

    # def compute_unlabeled_loss(self, net_output_list, sample, reduce=True):
    #     select_atom = sample["net_input"]["select_atom"]
    #     target = sample["target"]["finetune_target"]
    #     net_output_list = [torch.sort(net_output_list[i].squeeze(1))[0] for i in range(len(net_output_list))]
    #     target_list = [torch.sort(target[i][select_atom[i]==1])[0] for i in range(len(target))]

    #     if self.args.T == 0:
    #         weight_list = [torch.ones_like(net_output_list[i]) for i in range(len(net_output_list))]
    #     else:
    #         with torch.no_grad():
    #             weight_list = [get_confidence(net_output_list[i], target_list[i], self.args) for i in range(len(net_output_list))]
        
    #     loss = F.mse_loss(    # l1_loss mse_loss
    #         torch.concat(net_output_list),
    #         torch.concat(target_list),
    #         reduction="none",
    #     )
        
    #     weighted_loss = loss * torch.concat(weight_list).view(-1, 1).view(-1, 1)  # 确保 weights 的形状可以广播
    #     final_loss = weighted_loss.mean()
    #     return final_loss, torch.concat(weight_list).flatten()

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        num_tasks = logging_outputs[0]["num_task"]
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / num_tasks, sample_size, round=3
        )
        def reg_metrics(targets, predicts):
            r2 = r2_score(targets, predicts)
            mae = mean_absolute_error(targets, predicts)
            mse = mean_squared_error(targets, predicts)
            rmse = math.sqrt(mse)
            return r2, mae, mse, rmse

        if 'valid' in split or 'test' in split:
            predicts = np.concatenate([log.get("predict") for log in logging_outputs], axis=0)
            # predicts = predicts.detach().cpu().numpy()
            targets = np.concatenate([log.get("target") for log in logging_outputs], axis=0)
            # targets = targets.detach().cpu().numpy()
            ##### 
            r2, mae, mse, rmse = reg_metrics(targets, predicts)
            metrics.log_scalar("{}_r2".format(split), r2, sample_size, round=4)
            metrics.log_scalar("{}_mae".format(split), mae, sample_size, round=4)
            metrics.log_scalar("{}_mse".format(split), mse, sample_size, round=4)
            metrics.log_scalar("{}_rmse".format(split), rmse, sample_size, round=4)
            #####
            src_tokens = torch.cat([log.get("src_token")[log.get("select_atom")==1] for log in logging_outputs], dim=0)
            src_tokens = src_tokens.detach().cpu().numpy()
            elemenets = set(src_tokens)
            if len(elemenets) > 1:
                for element in elemenets:
                    element_targets = targets[src_tokens==element]
                    element_predicts = predicts[src_tokens==element]
                    r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                    element_sample_size = len(src_tokens[src_tokens==element])
                    metrics.log_scalar("{}_{}_r2".format(split, [element]), r2, element_sample_size, round=4)
                    metrics.log_scalar("{}_{}_mae".format(split, [element]), mae, element_sample_size, round=4)
                    metrics.log_scalar("{}_{}_mse".format(split, [element]), mse, element_sample_size, round=4)
                    metrics.log_scalar("{}_{}_rmse".format(split, [element]), rmse, element_sample_size, round=4)
            # unlabeled_loss_sum = sum(log.get("unlabeled_loss", 0) for log in logging_outputs)
            # metrics.log_scalar("unlabeled_loss", unlabeled_loss_sum / sample_size, sample_size, round=3)
    
@register_loss("atom_regloss_mae")   
class AtomRegMAELoss(AtomRegMSELoss):
    def __init__(self, task):
        super().__init__(task)
        # self.target_scaler = TargetScaler()
        self.target_scaler = task.target_scaler

    def compute_loss(self, net_output, sample, reduce=True):
        select_atom = sample["net_input"]["select_atom"].view(-1, 1)
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = sample['target']['finetune_target'].view(-1, self.args.num_classes).float()
        # print("loss.py", predicts, targets)
        # normalize_targets = torch.from_numpy(self.target_scaler.transform(targets.cpu())).to(targets.device).float()
        loss = F.l1_loss(    # l1_loss mse_loss
            predicts,
            targets[select_atom==1].view(-1, self.args.num_classes),
            # reduction="sum" if reduce else "none",
        )
        # l1_loss = nn.L1Loss()
        # loss = l1_loss(    # l1_loss mse_loss
        #     predicts[select_atom==1],
        #     normalize_targets[select_atom==1],
        # )
        residues = (predicts - targets[select_atom==1].view(-1, self.args.num_classes)).data
        return loss, residues
    
    def compute_unlabeled_loss(self, net_output_list, sample, reduce=True):
        select_atom = sample["net_input"]["select_atom"]
        target = sample["target"]["finetune_target"]
        net_output_list = [torch.sort(net_output_list[i].squeeze(1))[0] for i in range(len(net_output_list))]
        target_list = [torch.sort(target[i][select_atom[i]==1])[0] for i in range(len(target))]
        
        # if self.args.T == 0:
        weight_list = [torch.ones_like(net_output_list[i]) for i in range(len(net_output_list))]
            # weight_list = [torch.ones_like(net_output_list[i]) / len(net_output_list[i]) for i in range(len(net_output_list))]
        weights = torch.concat(weight_list).view(-1, 1).view(-1, 1)
            # weights /= weights.mean()
        # elif self.args.p_scale == None:
        #     weight_list = [torch.zeros_like(net_output_list[i]) for i in range(len(net_output_list))]
        #     weights = torch.concat(weight_list).view(-1, 1).view(-1, 1)
        # else:
        #     with torch.no_grad():
        #         weight_list = [get_confidence(net_output_list[i], target_list[i], self.args) for i in range(len(net_output_list))]
        #         weights = torch.concat(weight_list).view(-1, 1).view(-1, 1)
        #         weights /= weights.mean()
        
        losses = F.l1_loss(    # l1_loss mse_loss
            torch.concat(net_output_list),
            torch.concat(target_list),
            reduction="none",
        )
        
        weighted_loss = losses * weights
        final_loss = weighted_loss.mean()
        # soft loss
        # final_loss = [l2_loss(net_output_list[i], target_list[i], sigma=self.args.sigma) for i in range(len(net_output_list))]
        # final_loss = torch.stack(final_loss).mean()
        return final_loss, weights, losses.data

# def inner_product(tensor1, tensor2, sigma=1.0):
#     """Compute similarity between two lists using analytic Gaussian convolution."""
#     diff = tensor1.unsqueeze(0) - tensor2.unsqueeze(1)  # Pairwise differences
#     matrix = torch.exp(-0.5 * (diff / sigma) ** 2)
#     # matrix = - torch.abs(diff)
#     return matrix.sum()

# def l2_loss(tensor1, tensor2, sigma=1.0):
#     return (inner_product(tensor1, tensor1, sigma=sigma) + inner_product(tensor2, tensor2, sigma=sigma) - 2 * inner_product(tensor1, tensor2, sigma=sigma)) / 2

# def get_confidence(output, target, args):
#     if args.p_dist == 'cauchy':
#         log_p = lambda x: torch.distributions.Cauchy(loc=0, scale=args.p_scale).log_prob(x)
#     elif args.p_dist == 'laplace':
#         log_p = lambda x: torch.distributions.Laplace(loc=0, scale=args.p_scale).log_prob(x)
#     elif args.p_dist == 'gaussian':
#         log_p = lambda x: torch.distributions.Normal(loc=0, scale=args.p_scale).log_prob(x)
#     else:
#         raise ValueError(f"Unsupported distribution: {args.p_dist}")
    
#     # pre-compute
#     log_p_cache = log_p(output.view(-1, 1) - target.view(1, -1))
    
#     match_tol = 0
    
#     # Pre-compute indices for searchsorted
#     left_indices = torch.searchsorted(target, target - match_tol) - 1
#     right_indices = torch.searchsorted(target, target + match_tol, right=True)

#     # Compute diagonal terms for log_p_cache
#     log_p_diag = log_p_cache.diagonal()
#     log_p_sys = log_p_cache + log_p_cache.T

#     # Compute diff_left
#     valid_left_mask = left_indices >= 0
#     left_indices = left_indices.clamp(0, log_p_sys.size(1) - 1)
#     left_term_1 = log_p_diag + log_p_diag[left_indices]
#     left_term_2 = log_p_sys.gather(1, left_indices.unsqueeze(1)).squeeze(1)
#     diff_left = torch.where(valid_left_mask, left_term_1 - left_term_2, float('inf'))

#     # Compute diff_right
#     valid_right_mask = right_indices < len(target)
#     right_indices = right_indices.clamp(0, log_p_sys.size(1) - 1)
#     right_term_1 = log_p_diag + log_p_diag[right_indices]
#     right_term_2 = log_p_sys.gather(0, right_indices.unsqueeze(0)).squeeze(0)
#     diff_right = torch.where(valid_right_mask, right_term_1 - right_term_2, float('inf'))
        
#     # Compute confidence
#     # confidence = torch.max(torch.tensor(0.0, device=target.device), 
#     #                    1 - torch.exp(-diff_left / args.T) - torch.exp(-diff_right / args.T))
#     # Consider bad data confidence
#     confidence = torch.exp((log_p_diag - log_p(torch.tensor(0.0, device=target.device))) / args.T)
#     return confidence

