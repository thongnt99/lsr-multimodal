from torch import nn
from transformers import AutoModelForMaskedLM, AutoModel, AutoConfig, PretrainedConfig, PreTrainedModel
import torch


class D2SConfig(PretrainedConfig):
    model_type = "d2s"

    def __init__(self, mlm_head="distilbert-based-uncased", dense_size=256, **kwargs):
        self.mlm_head = mlm_head
        self.dense_size = dense_size
        super().__init__(**kwargs)


class D2SModel(PreTrainedModel):
    config_class = D2SConfig

    def __init__(self, config: D2SConfig = D2SConfig()):
        super().__init__(config)
        model = AutoModelForMaskedLM.from_pretrained(config.mlm_head)
        self.proj = nn.Linear(config.dense_size, model.config.hidden_size)
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


AutoConfig.register("d2s", D2SConfig)
AutoModel.register(D2SConfig, D2SModel)
