import torch
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx+self.sequence_length]
        target_ids = self.tokens[idx+1:idx+self.sequence_length+1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)