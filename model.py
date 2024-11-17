import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        # Add embedding layer for both source and target
        src = self.embedding(src)  # Shape: (batch_size, seq_len, d_model)
        tgt = self.embedding(tgt)  # Shape: (batch_size, seq_len, d_model)
        
        # Pass through transformer
        output = self.transformer(src, tgt)
        
        # Output shape: (batch_size, seq_len, vocab_size)
        output = self.fc_out(output)
        
        return output
