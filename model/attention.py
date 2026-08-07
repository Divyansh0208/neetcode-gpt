import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()

        torch.manual_seed(0)

        # Order matters for reproducible weights
        self.key = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.query = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:

        Q = self.query(embedded)
        K = self.key(embedded)
        V = self.value(embedded)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))

        context_length = embedded.shape[1]

        mask = torch.tril(torch.ones(context_length, context_length)).to(embedded.device)

        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=2)

        output = torch.matmul(attention_weights, V)


        return torch.round(output, decimals=4)