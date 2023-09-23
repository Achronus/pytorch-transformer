import torch
import torch.nn as nn

from tqdm import tqdm

from model.transformer import Transformer


if __name__ == '__main__':
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    batch_size = 128
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 1500
    dropout = 0.1
    epochs = 10

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                              num_layers, d_ff, max_seq_length, dropout,
                              device=device)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for i, epoch in tqdm(enumerate(range(epochs), 1)):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data)
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {i}, Loss: {loss.item()}")
