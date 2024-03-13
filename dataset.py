import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class DNADataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_excel(file_path)
        self.sequences = data["sequence\n"].values
        self.scores = data["score\n( %)"].values


    def __len__(self):
        return len(self.sequences)

    def one_hot_encode(self, sequence):
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoding = np.zeros((len(sequence), len(base_dict)))
        for i, base in enumerate(sequence):
            encoding[i, base_dict[base]] = 1
        return encoding


    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_sequence = self.one_hot_encode(sequence)
        encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.float32)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return encoded_sequence, score

def MyDataLoader(file_path, batch_size=1, shuffle=False, num_workers=2):
    dataset = DNADataset(file_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader

#
train_dataset = MyDataLoader("train.xlsx")

if __name__ == '__main__':
    for batch in train_dataset:
        encoding, score = batch
        print("Encoding:", encoding)
        print("Score:", score)
        break

