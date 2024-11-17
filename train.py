import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tiktoken
from model import TransformerModel
from dataset import TextDataset

with open('shakespeare.txt', 'r') as f:
    text = f.read()

# Set up encoding and tokenizer
encoder = tiktoken.get_encoding("gpt2")

# Tokenize the text
tokens = encoder.encode(text)

sequence_length = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create input-output pairs
input_ids = []
target_ids = []

for i in range(len(tokens) - sequence_length):
    input_ids.append(tokens[i:i+sequence_length])
    target_ids.append(tokens[i+1:i+sequence_length+1])  # next token as target

# Dataset and DataLoader setup
dataset = TextDataset(input_ids, target_ids)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = TransformerModel(vocab_size=encoder.n_vocab)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for input_batch, target_batch in dataloader:
        optimizer.zero_grad()
        
        # Move to device (GPU if available)
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        # Forward pass
        # Exclude the last token from target for next token prediction
        output = model(input_batch, target_batch)  # Pass both input and target directly
        
        # Calculate loss - flatten the tensors
        loss = criterion(output.view(-1, encoder.n_vocab), target_batch[:, 1:].contiguous().view(-1))  # Next word prediction
        
        # Backward pass
        loss.backward()
        optimizer.step()

        print("Loss is: " + str(loss))
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save the model state
with open('model_state.pt', 'wb') as f:
    torch.save(model.state_dict(), f)
