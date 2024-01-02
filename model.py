from torch import nn
from transformers import AutoModelForMaskedLM
import torch


class MLM(nn.Module):
    def __init__(self, dense_size):
        super().__init__()
        model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        self.proj = nn.Linear(dense_size, model.config.hidden_size)
        self.vocab_layer_norm = model.vocab_layer_norm
        self.vocab_projector = model.vocab_projector

    def forward(self, dense_vec, input_ids=None, attention_mask=None):
        dense_vec = self.proj(dense_vec)
        dense_vec = self.vocab_layer_norm(dense_vec)
        batch_size = len(dense_vec)
        term_importances = torch.log1p(
            torch.relu(self.vocab_projector(dense_vec)))
        if input_ids is not None and attention_mask is not None:
            mask = torch.zeros_like(term_importances).float()
            weights = torch.ones_like(input_ids).float()
            weights = weights * attention_mask.float()
            mask[torch.arange(batch_size).unsqueeze(-1).int(),
                 input_ids.int()] = weights
            term_importances = term_importances * mask
        return term_importances
