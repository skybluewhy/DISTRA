import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import os


alphabets = {
    "PAD": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}


def get_sample_id_list(l):
    sample_id_list = []
    sample_dict = dict()
    for m in l:
        if m not in sample_dict:
            sample_dict[m] = len(sample_dict)
        sample_id_list.append(sample_dict[m])
    return sample_id_list


def get_clonotype_list(l):
    clonotype_list = []
    clonotype_dict = dict()
    for m in l:
        if m not in clonotype_dict:
            clonotype_dict[m] = len(clonotype_dict)
        clonotype_list.append(clonotype_dict[m])
    return clonotype_list


def get_cell_type_list(l):
    cell_type_list = []
    cell_type_dict = dict()
    cell_type_cnt = dict()
    for m in l:
        if m not in cell_type_dict:
            cell_type_dict[m] = len(cell_type_dict)
            cell_type_cnt[m] = 0
        cell_type_cnt[m] += 1
        cell_type_list.append(cell_type_dict[m])
    return cell_type_list


def get_binding_name_list(l, cell_type_list):
    binding_name_list = []
    binding_name_dict = dict()
    binding_name_cnt = dict()
    binding_name_to_cell_type_cnt = dict()
    cnt = 0
    for m in l:
        if m not in binding_name_dict:
            binding_name_dict[m] = len(binding_name_dict)
            binding_name_cnt[m] = 0
        binding_name_cnt[m] += 1
        binding_name_list.append(binding_name_dict[m])

        if m not in binding_name_to_cell_type_cnt:
            binding_name_to_cell_type_cnt[m] = {}
        if cell_type_list[cnt] not in binding_name_to_cell_type_cnt[m]:
            binding_name_to_cell_type_cnt[m][cell_type_list[cnt]] = 0
        binding_name_to_cell_type_cnt[m][cell_type_list[cnt]] += 1
        cnt += 1
    return binding_name_list


import random
def get_split_label_list(num: int, train_ratio=0.9, seed=42):
    random.seed(seed)
    # 计算数量
    train_num = int(num * train_ratio)
    test_num = num - train_num
    # 构造标签
    label_list = ["train"] * train_num + ["test"] * test_num
    # 打乱
    random.shuffle(label_list)
    return label_list


class my_dataset(Dataset):
    def __init__(self, gex_file, split="None", random_seed=42):
        data = sc.read_h5ad(gex_file)
        data.obs['donor'] = get_sample_id_list(data.obs['donor'].tolist())
        data.obs['clonotype'] = get_clonotype_list(data.obs['clonotype'].tolist())
        data.obs['binding_name'] = get_binding_name_list(data.obs['binding_name'].tolist(), data.obs['cell_type'].tolist())
        data.obs['cell_type'] = get_cell_type_list(data.obs['cell_type'].tolist())

        if split == "train" or split == "test":
            data.obs['random_split'] = get_split_label_list(len(data.obs['donor'].tolist()), seed=random_seed)
            keep_mask = data.obs.random_split == split
            data = data[keep_mask]

        self.gex_profile = data.X

        self.donor = data.obs['donor'].tolist()
        self.clonotype = data.obs['clonotype'].tolist()
        self.cell_type = data.obs['cell_type'].tolist()

        self.beta_chains = list(data.obs.cdr3)
        n = 0
        self.TCR_ids = {}
        for idx, beta in enumerate(self.beta_chains):
            if beta not in self.TCR_ids:
                self.TCR_ids[beta] = n
                n += 1

    def aa_index(self, tcr, max_len=30):
        tcr_index = []
        tcr_index_no_pad = []
        tcr_att = []
        for i in range(max_len):
            if i < len(tcr):
                tcr_index.append(alphabets[tcr[i]])
                tcr_index_no_pad.append(alphabets[tcr[i]])
                tcr_att.append(True)
            else:
                tcr_index.append(0)
                tcr_att.append(False)
        return tcr_index, tcr_index_no_pad, tcr_att

    def __getitem__(self, item):
        batch_profile = self.gex_profile[item].toarray().squeeze()
        # batch_profile = self.gex_profile[item].squeeze()
        batch_profile = torch.tensor(batch_profile, dtype=torch.float32)
        tcr_index, tcr_index_no_pad, tcr_att = self.aa_index(self.beta_chains[item])
        tcr_index, tcr_att = torch.LongTensor(tcr_index), torch.BoolTensor(tcr_att)
        return batch_profile, self.TCR_ids[self.beta_chains[item]], tcr_index, tcr_att, self.donor[item], self.clonotype[item], self.cell_type[item], torch.FloatTensor([0 for i in range(6)])

    def __len__(self):
        return self.gex_profile.shape[0]
