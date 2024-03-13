import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class DNADataset(Dataset):
    def __init__(self, file_path, kmer_length=4):
        data = pd.read_excel(file_path)
        self.sequences = data["sequence\n"].values
        self.scores = data["score\n( %)"].values
        self.kmer_length = kmer_length

    def __len__(self):
        return len(self.sequences)

    def one_hot_encode(self, sequence):
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoding = np.zeros((len(sequence), len(base_dict)))
        for i, base in enumerate(sequence):
            encoding[i, base_dict[base]] = 1
        return encoding

    def kmer_embedding(self, sequence):
        embeddings = []
        for i in range(len(sequence) - self.kmer_length + 1):
            kmer = sequence[i:i+self.kmer_length]
            embedding = self.one_hot_encode(kmer)
            embeddings.append(embedding.flatten())
        return np.array(embeddings)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # 使用k-mer嵌入
        embeddings = self.kmer_embedding(sequence)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)

        return embeddings, score

def MyDataLoader(file_path, batch_size=1, shuffle=False, num_workers=2, kmer_length=4):
    dataset = DNADataset(file_path, kmer_length=kmer_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


train_dataset = MyDataLoader("train.xlsx")

if __name__ == '__main__':
    for batch in train_dataset:
        encoding, score = batch
        print("Encoding:", encoding)
        print("Score:", score)
        break
