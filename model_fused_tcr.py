import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.kl import kl_divergence
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.special import gammaln
from torch.nn.functional import gaussian_nll_loss
from transformer import TransformerEncoder


def log_nb_positive(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        eps: float = 1e-8,
        log_fn: callable = torch.log,
        lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
            theta * (log(theta + eps) - log_theta_mu_eps)
            + x * (log(mu + eps) - log_theta_mu_eps)
            + lgamma(x + theta)
            - lgamma(theta)
            - lgamma(x + 1)
    )

    return res


# RNA专用重建损失
def rna_recon_loss(rna_x, mu_nb, theta_nb):
    ll = log_nb_positive(rna_x, mu_nb, theta_nb)
    nll = -ll.sum(dim=-1).mean()
    # nll = (-ll).mean(dim=-1).mean()
    return nll


# TCR序列token重建损失（忽略padding mask）
def tcr_token_recon_loss(tcr_logits, tcr_idx, tcr_mask):
    B, L, V = tcr_logits.shape
    # CrossEntropy输入要求 [B,V,L], target [B,L]
    ce = F.cross_entropy(
        tcr_logits.reshape(-1, V),
        tcr_idx.reshape(-1),
        reduction="none"
    )
    # mask屏蔽padding位置（mask=0为pad，不计损失）
    valid_mask = tcr_mask.bool().reshape(-1)
    valid_ce = ce[valid_mask]
    if valid_ce.numel() == 0:
        return torch.tensor(0.0, device=tcr_logits.device)
    return valid_ce.mean()


def compute_clone_contrastive_loss(c_emb, clone_index, temperature=0.07):
    device = c_emb.device
    bs = c_emb.size(0)
    cr = F.normalize(c_emb, dim=1)

    # similarity matrix
    sim = torch.matmul(cr, cr.T) / temperature
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()

    # mask
    pos_mask = (clone_index.unsqueeze(0) == clone_index.unsqueeze(1)).float()
    self_mask = torch.eye(bs, device=device)
    pos_mask = pos_mask * (1.0 - self_mask)  # exclude self from positives
    all_mask = 1.0 - self_mask               # exclude self from denominator

    exp_logits = torch.exp(logits) * all_mask
    log_sum = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    log_prob = logits - log_sum

    pos_log_prob = (pos_mask * log_prob).sum(dim=1)
    pos_count = pos_mask.sum(dim=1)

    valid = (pos_count > 0.0).float()  # skip samples with zero positive in this batch
    mean_pos = pos_log_prob / (pos_count + 1e-8)

    # 关键：不要直接sum，除以有效样本数，loss量级稳定
    loss = - (valid * mean_pos).sum()
    num_valid = valid.sum()
    if num_valid > 0:
        loss = loss / num_valid
    else:
        loss = 0.0 * cr.sum()  # 保证still has grad

    return loss


def reparametrize(mu, logvar, current_device):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(current_device)
    return mu + eps * std


class px(nn.Module):
    def __init__(
            self,
            hidden_layers,
            rna_dim,          # RNA基因维度
            zx_dim, zd_dim, zy_concat_dim,  # 所有隐变量拼接维度
            restrict_recon_pos=False,
    ):
        super().__init__()
        self.rna_dim = rna_dim
        self.restrict_recon_pos = restrict_recon_pos

        # 解码器升维网络
        hidden_dim_rev = hidden_layers[::-1]
        latent_concat_dim = zx_dim + zd_dim + zy_concat_dim
        modules = []
        curr_dim = latent_concat_dim
        for hdim in hidden_dim_rev:
            modules.append(nn.Sequential(nn.Linear(curr_dim, hdim), nn.ReLU()))
            curr_dim = hdim
        self.decoder_mlp = nn.Sequential(*modules)

        # 仅保留RNA NB输出头，彻底删除高斯TCR分支
        self.fc_mu_nb = nn.Linear(curr_dim, rna_dim)
        self.fc_theta_nb = nn.Linear(curr_dim, rna_dim)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, zx, zd, zy_concat):
        """输入全部隐变量拼接，仅输出RNA NB分布参数"""
        latent = torch.cat([zx, zd, zy_concat], dim=1)
        h = self.decoder_mlp(latent)
        mu_nb = F.softplus(self.fc_mu_nb(h))
        theta_nb = F.softplus(self.fc_theta_nb(h))
        # 不再返回高斯参数，只输出RNA NB
        # ========= 新增钳位，防止掉到0附近产生 lgamma Inf =========
        mu_nb = torch.clamp(mu_nb, min=1e-6)
        theta_nb = torch.clamp(theta_nb, min=1e-6)
        return mu_nb, theta_nb


class TCRTokenDecoder(nn.Module):
    def __init__(
        self,
        hidden_latent_dims,  # 隐变量输入升维层
        latent_total_dim,    # zx+zd+zy拼接维度
        output_size,
    ):
        super().__init__()
        layers = []
        curr = latent_total_dim
        for dim in hidden_latent_dims:
            layers.append(nn.Sequential(nn.Linear(curr, dim), nn.ReLU()))
            curr = dim
        self.mlp = nn.Sequential(*layers)
        # 输出：(batch, max_seq_len * vocab_size)
        self.head = nn.Linear(curr, output_size)
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, z_concat):
        """
        z_concat: [B, latent_total_dim] 全部隐变量拼接
        return token_logits: [B, max_seq_len, vocab_size]
        """
        B = z_concat.shape[0]
        h = self.mlp(z_concat)
        output = self.head(h)
        return output


class pzd(nn.Module):
    def __init__(self, hidden_layers, d_dim, zd_dim):
        super(pzd, self).__init__()
        self.emb = nn.Embedding(num_embeddings=d_dim, embedding_dim=hidden_layers[-1])

        # Reconstruction of d using hidden layers
        modules = []
        hidden_dim_rev = hidden_layers[::-1]
        latent_dim_concat = hidden_layers[-1]
        for hidden_dim in hidden_dim_rev:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_d = nn.Linear(latent_dim_concat, zd_dim)
        self.fc_logvar_d = nn.Linear(latent_dim_concat, zd_dim)

        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Define the decoder
        self.decoder_pzd = nn.Sequential(*modules)

    def forward(self, label_idx):
        emb_feat = self.emb(label_idx)  # [B, emb_dim]
        h = self.decoder_pzd(emb_feat)
        mu_zd = self.fc_mu_d(h)
        logvar_zd = self.fc_logvar_d(h)
        return mu_zd, logvar_zd


class pzy(nn.Module):
    def __init__(self, hidden_layers, y_dim, zy_dim):
        super(pzy, self).__init__()
        self.emb = nn.Embedding(num_embeddings=y_dim, embedding_dim=hidden_layers[-1])
        self.emb_linear = nn.Linear(y_dim, hidden_layers[-1])
        # Reconstruction of y using hidden layers
        modules = []
        hidden_dim_rev = hidden_layers[::-1]
        latent_dim_concat = hidden_layers[-1]
        for hidden_dim in hidden_dim_rev:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_y = nn.Linear(latent_dim_concat, zy_dim)
        self.fc_logvar_y = nn.Linear(latent_dim_concat, zy_dim)

        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Define the decoder
        self.decoder_pzy = nn.Sequential(*modules)

    def forward(self, label_idx):
        if label_idx.is_floating_point():
            emb_feat = self.emb_linear(label_idx)
        else:
            emb_feat = self.emb(label_idx)
        # emb_feat = self.emb(label_idx)  # [B, emb_dim]
        h = self.decoder_pzy(emb_feat)
        mu_zy = self.fc_mu_y(h)
        logvar_zy = self.fc_logvar_y(h)
        return mu_zy, logvar_zy


class qzx(nn.Module):
    def __init__(self, hidden_layers, x_dim, zx_dim):
        super(qzx, self).__init__()

        # Inference of zx using hidden layers
        modules = []
        latent_dim_concat = x_dim
        for hidden_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),  # Stabilize the training for large count values
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_zx = nn.Linear(latent_dim_concat, zx_dim)
        self.fc_logvar_zx = nn.Linear(latent_dim_concat, zx_dim)

        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Define the encoder
        self.encoder_qzx = nn.Sequential(*modules)

    def forward(self, x_total):
        h = self.encoder_qzx(x_total)
        mu_zx = self.fc_mu_zx(h)
        logvar_zx = self.fc_logvar_zx(h)
        return mu_zx, logvar_zx


class qzd(nn.Module):
    def __init__(self, hidden_layers, x_dim, zd_dim):
        super(qzd, self).__init__()

        # Inference of zd using hidden layers
        modules = []
        latent_dim_concat = x_dim
        for hidden_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),  # Stabilize the training for large count values
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_zd = nn.Linear(latent_dim_concat, zd_dim)
        self.fc_logvar_zd = nn.Linear(latent_dim_concat, zd_dim)

        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Define the encoder
        self.encoder_qzd = nn.Sequential(*modules)

    def forward(self, x_total):
        h = self.encoder_qzd(x_total)
        mu_zd = self.fc_mu_zd(h)
        logvar_zd = self.fc_logvar_zd(h)
        return mu_zd, logvar_zd


class qzy(nn.Module):
    def __init__(self, hidden_layers, x_dim, zy_dim):
        super(qzy, self).__init__()

        # Inference of zy using hidden layers
        modules = []
        latent_dim_concat = x_dim
        for hidden_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim_concat, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),  # Stabilize the training for large count values
                    nn.ReLU()
                )
            )
            latent_dim_concat = hidden_dim
        self.fc_mu_zy = nn.Linear(latent_dim_concat, zy_dim)
        self.fc_logvar_zy = nn.Linear(latent_dim_concat, zy_dim)

        # Xavier initialization for all modules
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Define the encoder
        self.encoder_qzy = nn.Sequential(*modules)

    def forward(self, x_total):
        h = self.encoder_qzy(x_total)
        mu_zy = self.fc_mu_zy(h)
        logvar_zy = self.fc_logvar_zy(h)
        return mu_zy, logvar_zy


class qd(nn.Module):
    def __init__(self, d_dim, zd_dim):
        super(qd, self).__init__()

        # Prediction of d using single linear layer
        self.fc1 = nn.Linear(zd_dim, d_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d


class qy(nn.Module):
    def __init__(self, y_dim, zy_dim):
        super(qy, self).__init__()

        # Prediction of y using single linear layer
        self.fc1 = nn.Linear(zy_dim, y_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)
        return loc_y


class TR_model(nn.Module):
    def __init__(
            self,
            rna_dim: int,
            tcr_seq_emb_dim: int,
            y_dims: list,  # [subcell_num, tcr_emb_dim, expansion=2]
            d_dim: int,    # sampleID独热编码维度
            zx_dim: int, zd_dim: int, zy_dims: list,
            hidden_layers_x: list, hidden_layers_y: list, hidden_layers_d: list,
            num_y_covars=1,
            y_covar_space=["discrete"],
            y_dists=["categorical"],
            d_dist="categorical"
    ):
        super().__init__()
        self.rna_dim = rna_dim
        self.tcr_seq_emb_dim = tcr_seq_emb_dim
        self.total_x_dim = rna_dim + tcr_seq_emb_dim
        self.y_dims = y_dims
        self.d_dim = d_dim
        self.zx_dim = zx_dim
        self.zd_dim = zd_dim
        self.zy_dims = zy_dims
        self.num_y_covars = num_y_covars
        self.y_covar_space = y_covar_space
        self.y_dists = y_dists
        self.d_dist = d_dist

        # 1. TCR序列专用编码器
        tcr_params = {
            'embedding_size': 32,
            'num_heads': 4,
            'forward_expansion': 4,
            'encoding_layers': 3,
            'decoding_layers': 3,
            'dropout': 0.2,
            'beta_only': True,
            'max_tcr_length': 30
        }
        self.tcr_encoder = TransformerEncoder(tcr_params, tcr_seq_emb_dim, 30)

        # 2. 多模态解码器（RNA NB + TCR高斯）
        zy_sum = sum(zy_dims)
        self.px = px(
            hidden_layers=hidden_layers_x,
            rna_dim=rna_dim,
            zx_dim=zx_dim, zd_dim=zd_dim, zy_concat_dim=zy_sum)

        # 3. 先验网络：pzd(病人批次) + pzy三套生物学先验
        self.pzd = pzd(hidden_layers_d, d_dim, zd_dim)
        self.pzy = nn.ModuleList()
        for i in range(num_y_covars):
            self.pzy.append(pzy(hidden_layers_y, y_dims[i], zy_dims[i]))

        # 4. 后验编码器
        self.qzx = qzx(hidden_layers_x, self.rna_dim, zx_dim)
        self.qzd = qzd(hidden_layers_x, self.rna_dim, zd_dim)
        self.qzy1 = qzy(hidden_layers_x, self.rna_dim, zy_dims[0])
        self.qzy2 = qzy(hidden_layers_x, self.rna_dim, zy_dims[1])

        # 5. 辅助分类头
        self.qd = qd(d_dim, zd_dim)
        self.qy = nn.ModuleList([
            qy(y_dims[i], zy_dims[i]) for i in range(num_y_covars)
        ])
        self.clono_proj = nn.Sequential(
            nn.Linear(zy_dims[1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        self.tcr_encoder_proj = nn.Linear(768, zy_dims[1])
        self.clono_proj_for_fused = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        # 只保留一个投影头，用于融合对比损失
        self.fused_proj1 = nn.Sequential(
            nn.Linear(zy_dims[1] + zy_dims[1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )

    def forward(self, rna_x, tcr_idx, tcr_mask, d, y_labels):
        """
        输入：
        rna_x: [B, 2000] RNA计数
        tcr_idx: [B, seq_len] TCR token索引
        tcr_mask: [B, seq_len] attention mask (0=pad)
        d_onehot: [B, d_dim] sampleID独热批次标签
        y_labels: list[3] 三套生物学标签
            y0: sub_cell_type [B] → onehot后输入pzy[0]
            y1: tcr_global_emb [B, emb_dim] TCR序列全局表征标签
            y2: expansion [B] 0/1
        """
        # Step2：全部后验编码器推断隐变量均值方差
        qzx_loc, qzx_logvar = self.qzx(rna_x)
        qzd_loc, qzd_logvar = self.qzd(rna_x)
        m1, lv1 = self.qzy1(rna_x)
        m2, lv2 = self.qzy2(rna_x)
        qzy_locs, qzy_logvars = [m1, m2], [lv1, lv2]

        # Step3：重参数采样所有Z
        zx = reparametrize(qzx_loc, qzx_logvar, rna_x.device)
        zd = reparametrize(qzd_loc, qzd_logvar, rna_x.device)
        zys = [reparametrize(m, lv, rna_x.device) for m, lv in zip(qzy_locs, qzy_logvars)]
        zy_concat = torch.cat(zys, dim=1)

        # Step4：解码器重构RNA+TCR分布参数
        mu_nb, theta_nb = self.px(zx, zd, zy_concat)

        # Step5：先验网络，从标签L生成先验分布
        pzd_loc, pzd_logvar = self.pzd(d)
        pzy_locs, pzy_logvars = [], []
        for i in range(self.num_y_covars):
            y = y_labels[i]
            pm, plv = self.pzy[i](y)
            pzy_locs.append(pm)
            pzy_logvars.append(plv)

        # Step6：辅助分类预测（从隐变量还原原始标签）
        qd_pred = self.qd(zd)
        qy_preds = [self.qy[i](zys[i]) for i in range(self.num_y_covars)]

        return (
            mu_nb, theta_nb, None,
            pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars,
            qd_pred, qy_preds, zx, zd, zys
        )

    # 获取后验隐变量（PCA/UMAP下游绘图）
    def get_posterior(self, rna_x, tcr_idx, tcr_mask, means=False):
        qzx_loc, qzx_logvar = self.qzx(rna_x)
        qzd_loc, qzd_logvar = self.qzd(rna_x)
        m1, lv1 = self.qzy1(rna_x)
        m2, lv2 = self.qzy2(rna_x)
        qzy_locs, qzy_logvars = [m1, m2], [lv1, lv2]

        zx = reparametrize(qzx_loc, qzx_logvar, rna_x.device)
        zd = reparametrize(qzd_loc, qzd_logvar, rna_x.device)
        zys = [reparametrize(m, lv, rna_x.device) for m, lv in zip(qzy_locs, qzy_logvars)]
        return zx, zys, zd

    def get_fused_emb(self, rna_x, tcr_idx):
        tcr_emb = self.tcr_encoder(tcr_idx)
        m1, lv1 = self.qzy1(rna_x)
        m2, lv2 = self.qzy2(rna_x)
        qzy_locs, qzy_logvars = [m1, m2], [lv1, lv2]
        zys = [reparametrize(m, lv, rna_x.device) for m, lv in zip(qzy_locs, qzy_logvars)]
        x_total = torch.cat([zys[1], tcr_emb], dim=1)
        fused_feat = self.fused_proj(x_total)
        return fused_feat

    # 条件生成：固定Zy1/Zy2/Zy3/Zd/Zr中部分因子，生成RNA+TCR表达（对应原文4.3）
    def conditional_generate(self, zx_fix, zd_fix, zys_fixed_list):
        zy_concat = torch.cat(zys_fixed_list, dim=1)
        mu_nb, theta_nb = self.px(zx_fix, zd_fix, zy_concat)
        return

    def get_fused_feat(self, tcr_enc_outputs, zys):
        tcr_enc_outputs = tcr_enc_outputs.flatten(1)
        tcr_emb = self.tcr_encoder_proj(tcr_enc_outputs)
        x_total = torch.cat([zys[1].detach(), tcr_emb], dim=1)
        fused_feat = self.fused_proj1(x_total)
        return fused_feat

    # 完整ELBO损失计算
    def loss(
            self, rna_x, zys, mu_nb, theta_nb, tcr_enc_outputs, clonotype_ids,
            pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars,
            qd_pred, qy_preds, d, y_labels,
            zys_beta_kl, zys_beta_aux,
            beta_rna=1.0, beta_tcr=1.0, beta_contrast=0.1, tau=0.1, device=None
    ):
        # 1. 重构损失 RNA NB
        recon_rna = rna_recon_loss(rna_x, mu_nb, theta_nb)

        # 2. KL损失 Zr 标准正态先验
        kl_zx = kl_divergence(
            dist.Normal(qzx_loc, torch.exp(0.5 * qzx_logvar)),
            dist.Normal(torch.zeros_like(qzx_loc), torch.ones_like(qzx_loc))
        ).sum(dim=1).mean()

        # 3. Zy KL损失（标签先验约束）
        kl_zys = []
        # 对比学习的空间仅做标准正态
        for i in range(self.num_y_covars):
            if self.y_dists[i] == "contrastive":
                kl = kl_divergence(
                    dist.Normal(qzy_locs[i], torch.exp(0.5 * qzy_logvars[i])),
                    dist.Normal(torch.zeros_like(qzy_locs[i]), torch.ones_like(qzy_locs[i]))
                ).sum(dim=1).mean()
            else:
                kl = kl_divergence(
                    dist.Normal(qzy_locs[i], torch.exp(0.5 * qzy_logvars[i])),
                    dist.Normal(pzy_locs[i], torch.exp(0.5 * pzy_logvars[i]))
                ).sum(dim=1).mean()
            kl_zys.append(kl * zys_beta_kl[i])
        kl_zys_total = torch.stack(kl_zys).mean(dim=0)

        # 4. Zd批次KL
        kl_zd = kl_divergence(
            dist.Normal(qzd_loc, torch.exp(0.5 * qzd_logvar)),
            dist.Normal(pzd_loc, torch.exp(0.5 * pzd_logvar))
        ).sum(dim=1).mean()

        # 5. 辅助分类损失
        # 批次sampleID损失
        if self.d_dist == "categorical":
            loss_d = F.cross_entropy(qd_pred, d)
        else:
            loss_d = F.binary_cross_entropy_with_logits(qd_pred, d)

        # 生物学标签损失
        loss_ys = []
        for i in range(self.num_y_covars):
            pred = qy_preds[i]
            y_true = y_labels[i]
            space = self.y_covar_space[i]
            dist_type = self.y_dists[i]
            if space == "discrete" and dist_type == "categorical":
                loss_y = F.cross_entropy(pred, y_true.long())
            elif space == "discrete" and dist_type == "bernoulli":
                loss_y = F.cross_entropy(pred, y_true.long())
            elif space == "continuous" and dist_type == "none":
                loss_y = F.mse_loss(pred, y_true, reduction="none")
                loss_y = loss_y.mean()
            elif space == "continuous" and dist_type == "contrastive":
                tcr_enc_outputs = tcr_enc_outputs.flatten(1)
                tcr_emb = self.tcr_encoder_proj(tcr_enc_outputs)
                x_total = torch.cat([zys[i].detach(), tcr_emb], dim=1)
                fused_feat = self.fused_proj1(x_total)
                # 2. 微观克隆约束 (同克隆进一步紧密聚集)
                clono_feat = self.clono_proj_for_fused(fused_feat)
                loss_clone = compute_clone_contrastive_loss(clono_feat, clonotype_ids)
                loss_y = loss_clone
                return loss_y
            else:
                continue
            loss_ys.append(loss_y * zys_beta_aux[i])
        loss_y_total = torch.stack(loss_ys).mean(dim=0)

        return recon_rna, 0, kl_zx, kl_zys_total, kl_zd, loss_d, loss_y_total, 0

