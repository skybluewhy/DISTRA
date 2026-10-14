import argparse
import torch
import numpy as np
import random
import os
from data_10X import my_dataset
import scanpy as sc
import anndata as ad
from model import TR_model


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


def test(args):
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
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = my_dataset(args.dataset, split="none", random_seed=args.seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    Multimodal_embedding = []

    # batch_profile, self.TCR_ids[self.beta_chains[item]], tcr_index, tcr_att, self.donor[item], self.clonotype[item], self.zy_immune_target[item]
    with torch.no_grad():
        for idx, b in enumerate(dataloader):
            zx, zys, zd = model.get_posterior(
                b[0].to(device),
                b[2].to(device),
                b[3].to(device),
            )
            zys = torch.cat(zys, dim=1)
            z = torch.cat((zd, zx, zys), dim=1)
            # fused_feat = model.get_fused_emb(b[0].to(device), b[2].to(device))
            Multimodal_embedding.append(z.to(torch.device("cpu")))

    original_data = sc.read_h5ad(args.dataset)
    Multimodal_embedding = torch.cat(Multimodal_embedding, dim=0)
    adata_multimodal = ad.AnnData(Multimodal_embedding.numpy())
    adata_multimodal.obs = original_data.obs
    adata_multimodal.write("z_emb.h5ad")


if __name__ == '__main__':
    args = parser()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    test(args)
