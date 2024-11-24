import torch
from model import LSTMModel
import tiktoken
import config

encoder = tiktoken.get_encoding("gpt2")

vocab_size = encoder.n_vocab
model = LSTMModel(vocab_size, config.embedding_size, config.hidden_size, config.num_layers).to(config.device)
model.load_state_dict(torch.load('models/shakespeare_lstm.pth', map_location=config.device))

def generate_text(model, start_text, max_length=100):
    model.eval()
    generated_tokens = encoder.encode(start_text)
    input_ids = torch.tensor(generated_tokens[-config.sequence_length:], dtype=torch.long).unsqueeze(0).to(config.device)

    hidden = model.init_hidden(1, config.hidden_size, config.num_layers, config.device)
    for _ in range(max_length):
        outputs, hidden = model(input_ids, hidden)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1).item()
        generated_tokens.append(next_token)

        # Prepare input for the next time step
        input_ids = torch.tensor([generated_tokens[-config.sequence_length:]], dtype=torch.long).to(config.device)

    return encoder.decode(generated_tokens)

# Example usage
start_text = "For who would bear the whips and scorns of time,Th' oppressor's"
generated_text = generate_text(model, start_text, max_length=100)
print(generated_text)