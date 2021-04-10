import torch
import torch.nn as nn
from reformer_pytorch import ReformerLM
from torch.nn import CrossEntropyLoss
from transformers.activations import get_activation

class ReformerNER(nn.Module):
  def __init__(self, dim, num_labels, hidden_dropout_prob=0.1):
    super().__init__()
    self.dense = nn.Linear(dim, 2*dim)
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.out_proj = nn.Linear(2*dim, num_labels)

  def forward(self, x, **kwargs):
    # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class ReformerNERModel(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads, num_labels=2, causal=False):
        super().__init__()
        self.reformer = ReformerLM(
                num_tokens=num_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                max_seq_len=max_seq_len,
                causal=causal,           # auto-regressive 학습을 위한 설정
                return_embeddings=True    # reformer 임베딩을 받기 위한 설정
            )
        self.ner = ReformerNER(dim, num_labels)
        self.num_labels = num_labels
    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                **kwargs):
        # 1. reformer의 출력
        outputs = self.reformer(input_ids, **kwargs)

        # 2. mrc를 위한
        logits = self.ner(outputs)
        loss_func = CrossEntropyLoss()
        #print(logits.view(-1, self.num_labels).shape)
        if labels is None:
            return logits
        if attention_mask is None:
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_func.ignore_index).type_as(labels)
            )
            loss = loss_func(active_logits, active_labels)
        return loss
