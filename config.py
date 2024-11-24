import torch

sequence_length = 256
batch_size = 32
hidden_size = 512
num_layers = 2
embedding_size = 128
learning_rate = 0.001
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")