import torch

sequence_length = 256
batch_size = 32
hidden_size = 512
num_layers = 4
embedding_size = 128
num_heads = 8
max_seq_length = sequence_length
learning_rate = 0.001
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")