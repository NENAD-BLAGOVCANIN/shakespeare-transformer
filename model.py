import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, hidden_size, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_size))
        self.transformer = nn.Transformer(
            d_model=embedding_size, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=hidden_size, 
            batch_first=True
        )
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, src, tgt):
        # Add positional encoding to the embeddings
        src_embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Transformer forward pass
        output = self.transformer(src_embedded, tgt_embedded)
        output = self.fc(output)
        return output
