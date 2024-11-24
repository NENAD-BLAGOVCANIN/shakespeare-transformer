import torch
from model import TransformerModel
import tiktoken
import config

encoder = tiktoken.get_encoding("gpt2")

vocab_size = encoder.n_vocab
model = TransformerModel(
    vocab_size=vocab_size,
    embedding_size=config.embedding_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    hidden_size=config.hidden_size,
    max_seq_length=config.max_seq_length,
).to(config.device)
model.load_state_dict(torch.load('models/transformer.pth', map_location=config.device))

def generate_text_transformer(model, start_text, max_length=100):
    model.eval()
    generated_tokens = encoder.encode(start_text)
    src = torch.tensor(generated_tokens[-config.sequence_length:], dtype=torch.long).unsqueeze(0).to(config.device)
    
    for _ in range(max_length):
        # Prepare tgt sequence (decoder input)
        tgt_input = torch.tensor([generated_tokens[-config.sequence_length:]], dtype=torch.long).to(config.device)

        # Generate the next token
        with torch.no_grad():
            outputs = model(src, tgt_input)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1).item()
            generated_tokens.append(next_token)

        # Update src for the next iteration if sequence grows longer
        if len(generated_tokens) > config.sequence_length:
            src = torch.tensor(generated_tokens[-config.sequence_length:], dtype=torch.long).unsqueeze(0).to(config.device)

    return encoder.decode(generated_tokens)

# Example usage
start_text = "To eat the world's due, by the grave and"
generated_text = generate_text_transformer(model, start_text, max_length=100)
print(generated_text)