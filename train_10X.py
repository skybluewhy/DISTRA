import argparse
import torch
import numpy as np
import random
import os
from data_10X import my_dataset
import scanpy as sc
import anndata as ad
from model import TR_model
import matplotlib.pyplot as plt


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = my_dataset(args.dataset, split="train", random_seed=args.seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    loss_weights = [1, 0.01, 0.01, 0.01, 1, 1, 1]

    loss_names = [
        "recon_rna",
        "recon_tcr",
        "kl_zx",
        "kl_zys_total",
        "kl_zd",
        "loss_d",
        "loss_y_total",
        "loss_contrast",
        "total_loss"
    ]
    loss_history = {k: [] for k in loss_names}
    iter_list = []
    global_iter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx, b in enumerate(dataloader):
            optimizer.zero_grad()

            mu_nb, theta_nb, tcr_output, pzd_loc, pzd_logvar, pzy_locs, pzy_logvars, \
            qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars, \
            qd_pred, qy_preds, zx, zd, zys = model(
                b[0].to(device),
                b[2].to(device),
                b[3].to(device),
                b[4].to(device),
                [b[6].to(device), b[7].to(device)]
            )

            recon_rna, recon_tcr, kl_zx, kl_zys_total, kl_zd, loss_d, loss_y_total, loss_contrast = model.loss(
                b[0].to(device), zys, mu_nb, theta_nb, b[2].to(device), b[5].to(device),
                pzd_loc, pzd_logvar, pzy_locs, pzy_logvars,
                qzx_loc, qzx_logvar, qzd_loc, qzd_logvar, qzy_locs, qzy_logvars,
                qd_pred, qy_preds, b[4].to(device), [b[6].to(device), b[7].to(device)],
                [1, 0.1], [1, 1]
            )

            loss = (0.1 * loss_weights[0] * recon_rna + loss_weights[0] * recon_tcr + loss_weights[1] * kl_zx + loss_weights[2] * kl_zys_total +
                    loss_weights[3] * kl_zd + loss_weights[4] * loss_d + loss_weights[5] * loss_y_total +
                    loss_weights[6] * loss_contrast)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if idx % 50 == 1:
                global_iter += 1
                iter_list.append(global_iter)
                loss_history["recon_rna"].append(recon_rna.item())
                loss_history["recon_tcr"].append(0)
                loss_history["kl_zx"].append(kl_zx.item())
                loss_history["kl_zys_total"].append(kl_zys_total.item())
                loss_history["kl_zd"].append(kl_zd.item())
                loss_history["loss_d"].append(loss_d.item())
                loss_history["loss_y_total"].append(loss_y_total.item())
                loss_history["loss_contrast"].append(0)
                loss_history["total_loss"].append(loss.item())

            # 每50个iter打印一次，避免刷屏
            if idx % 50 == 0:
                print(
                    f"Iter {idx:5d} | total_loss: {loss.item():.4f} | recon_rna: {recon_rna.item():.4f} | recon_tcr: {0:.4f}")

    # 绘图：横轴 iteration
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()
    plot_list = [
        ("recon_rna", 0),
        ("recon_tcr", 1),
        ("kl_zx", 2),
        ("kl_zys_total", 3),
        ("kl_zd", 4),
        ("loss_d", 5),
        ("loss_y_total", 6),
        ("loss_contrast", 7),
    ]
    for name, ax_idx in plot_list:
        ax = axes[ax_idx]
        ax.plot(iter_list, loss_history[name], linewidth=0.7)
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.save_dir, "loss_curves_iter.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nIter‑level loss curve saved: {fig_path}")

    np.save(os.path.join(args.save_dir, "loss_history_iter.npy"), {"iter": iter_list, "loss": loss_history})

    torch.save(model.state_dict(), args.save_dir + f"checkpoint_final.pt")


if __name__ == '__main__':
    args = parser()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args)
