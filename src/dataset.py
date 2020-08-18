from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from typing import List

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1

    return X


class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len

        seq_path = data_path / phase / 'global'
        self.seq_path = sorted(list(seq_path.glob('*')))
        self.max_seq_len = max_seq_len

        pkt_path = data_path / phase / 'pocket'
        self.pkt_path = sorted(list(pkt_path.glob('*')))
        self.max_pkt_len = max_pkt_len
        self.pkt_window = pkt_window
        self.pkt_stride = pkt_stride
        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset {phase}: will not fold pkt')

        assert len(self.seq_path) == len(self.pkt_path)
        assert len(self.seq_path) == len(self.smi)

        self.length = len(self.smi)

    def __getitem__(self, idx):
        seq = self.seq_path[idx]
        pkt = self.pkt_path[idx]
        assert seq.name == pkt.name

        _seq_tensor = pd.read_csv(seq, index_col=0).drop(['idx'], axis=1).values[:self.max_seq_len]
        seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
        seq_tensor[:len(_seq_tensor)] = _seq_tensor

        _pkt_tensor = pd.read_csv(pkt, index_col=0).drop(['idx'], axis=1).values[:self.max_pkt_len]
        if self.pkt_window is not None and self.pkt_stride is not None:
            pkt_len = (int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride))
                       * self.pkt_stride
                       + self.pkt_window)
            pkt_tensor = np.zeros((pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            pkt_tensor = np.array(
                [pkt_tensor[i * self.pkt_stride:i * self.pkt_stride + self.pkt_window]
                 for i in range(int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride)))]
            )
        else:
            pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor

        return (seq_tensor.astype(np.float32),
                pkt_tensor.astype(np.float32),
                label_smiles(self.smi[seq.name.split('.')[0]], self.max_smi_len),
                np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32))

    def __len__(self):
        return self.length
    