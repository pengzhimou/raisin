import torch
import torch.nn as nn
import pandas as pd

# Define the hyperparameters
d_model = 512
nhead = 8
num_layers = 6
dropout = 0.1

# Create the Transformer model
transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)

# Example input data (batch_size=2, seq_length=10, d_model=512)
src = torch.randn(10, 2, 512)
tgt = torch.randn(10, 2, 512)

# Pass the input data through the Transformer model
output = transformer(src, tgt)
output

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate dummy stock price data

def csv_to_Matrix(path): # https://www.delftstack.com/zh/howto/numpy/read-csv-to-numpy-array/
    x_Matrix = pd.read_csv(path, header=None)
    x_Matrix = np.array(x_Matrix)
    return x_Matrix


num_days = 200
stock_prices = np.random.rand(num_days) * 100

# Preprocess the data
input_seq_len = 10
output_seq_len = 5
num_samples = num_days - input_seq_len - output_seq_len + 1

src_data = torch.tensor([stock_prices[i:i+input_seq_len] for i in range(num_samples)]).unsqueeze(-1).float()
tgt_data = torch.tensor([stock_prices[i+input_seq_len:i+input_seq_len+output_seq_len] for i in range(num_samples)]).unsqueeze(-1).float()

# Create a custom Transformer model
class StockPriceTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(StockPriceTransformer, self).__init__()
        self.input_linear = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output

d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1

model = StockPriceTransformer(d_model, nhead, num_layers, dropout=dropout)

# Training parameters
epochs = 100
lr = 0.001
batch_size = 16

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for i in range(0, num_samples, batch_size):
        src_batch = src_data[i:i+batch_size].transpose(0, 1)
        tgt_batch = tgt_data[i:i+batch_size].transpose(0, 1)
        
        optimizer.zero_grad()
        output = model(src_batch, tgt_batch[:-1])
        loss = criterion(output, tgt_batch[1:])
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")



# # Predict the next 5 days of stock prices one at a time
# src = torch.tensor(stock_prices[-input_seq_len:]).unsqueeze(-1).unsqueeze(1).float()
# tgt = torch.zeros(output_seq_len, 1, 1)

# with torch.no_grad():
#     for i in range(output_seq_len):
#         prediction = model(src, tgt[:i+1])
#         tgt[i] = prediction[-1]

# output = tgt.squeeze().tolist()
# print("Next 5 days of stock prices:", output)




# Predict the next 5 days of stock prices at one time
src = torch.tensor(stock_prices[-input_seq_len:]).unsqueeze(-1).unsqueeze(1).float()
tgt = torch.zeros(output_seq_len, 1, 1)

with torch.no_grad():
     prediction = model(src, tgt)

output = prediction.squeeze().tolist()
print("Next 5 days of stock prices:", output)







