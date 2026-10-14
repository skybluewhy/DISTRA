import argparse
import torch
import numpy as np
import random
import os
from data_10X_fused_tcr import my_dataset
import scanpy as sc
import anndata as ad
from model_fused_tcr import TR_model
from transformers import BertModel


random_seed = 100
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)


def parser():
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', default=10, type=int)
    ap.add_argument('--batch_size', default=256, type=int)
    ap.add_argument('--lr', default=0.0001, type=float)
    ap.add_argument('--extract_feat', default=False, type=bool)
    ap.add_argument('--save_dir', default="./checkpoints/", type=str)
    ap.add_argument('--dataset', default="./data/10X/10X_high_appear_binding.h5ad", type=str)
    ap.add_argument('--load_model', default="./checkpoints/checkpoint10.pt", type=str)
    ap.add_argument('--save_feat', default="1", type=str)
    ap.add_argument('--seed', default=42, type=int)
    args = ap.parse_args()
    return args


# 单个基因集分别拆分分析
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rna_dim = 2000
    tcr_emb_dim = 10
    n_sample = 4
    model = TR_model(
        rna_dim=rna_dim,
        tcr_seq_emb_dim=tcr_emb_dim,
        y_dims=[5, 6],
        d_dim=n_sample,
        zx_dim=20, zd_dim=5, zy_dims=[20, 20],
        hidden_layers_x=[256, 128],
        hidden_layers_y=[128],
        hidden_layers_d=[64],
        num_y_covars=2,
        y_covar_space=["discrete", "continuous"],
        y_dists=["categorical", "contrastive"],
        d_dist="categorical"
    ).to(device)
    checkpoint = torch.load("./checkpoints/checkpoint_final.pt", map_location=torch.device(device))
    model.load_state_dict(checkpoint, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = my_dataset(args.dataset, split="train", random_seed=args.seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    tcr_encoder = BertModel.from_pretrained("./pretrain_model/tcr-bert").to(device)
    tcr_encoder.eval()

    # batch_profile, self.TCR_ids[self.beta_chains[item]], tcr_index, tcr_att, tcr_token_type_ids, self.donor[item], self.clonotype[item], self.cell_type[item], predict_prob_vec
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx, b in enumerate(dataloader):
            optimizer.zero_grad()

            with torch.no_grad():
                TCR_index_input = {"input_ids": b[2].to(device), "token_type_ids": b[4].to(device),
                                   "attention_mask": b[3].to(device)}
                tcr_enc_outputs = tcr_encoder(**TCR_index_input).last_hidden_state
                tcr_pooled = (tcr_enc_outputs * b[3].to(device).float().unsqueeze(-1)).sum(1) / (
                        b[3].to(device).float().sum(1, keepdim=True) + 1e-8)

            mu_nb, theta_nb, tcr_output, pzd_loc, pzd_logvar, pzy_locs, pzy_logvars, \
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars, \
            qd_pred, qy_preds, zx, zd, zys = model(
                b[0].to(device),
                b[2].to(device),
                b[3].to(device),
                b[5].to(device),
                [b[7].to(device), b[8].to(device)]
            )

            loss = model.loss(
                b[0].to(device), zys, mu_nb, theta_nb, tcr_pooled, b[6].to(device),
                pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
                qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars,
                qd_pred, qy_preds, b[5].to(device), [b[7].to(device), b[8].to(device)],
                [1, 0.1], [1, 1]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 每50个iter打印一次，避免刷屏
            if idx % 50 == 0:
                print(
                    f"Iter {idx:5d} | total_loss: {loss.item():.4f}")

    torch.save(model.state_dict(), args.save_dir + f"checkpoint_final_fused_tcr.pt")


if __name__ == '__main__':
    args = parser()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args)
