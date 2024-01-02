from torch import nn
from regularizer import L1
import torch


class BICELoss(nn.Module):
    def __init__(self, temp=0.001, q_reg=0.0, d_reg=0.0, T=1000,):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.q_regularizer = L1(q_reg, T)
        self.d_regularizer = L1(d_reg, T)
        self.temp = temp

    def forward(self, sparse_texts, sparse_imgs, dense_texts, dense_imgs):
        sparse_i2t_scores = sparse_imgs @ sparse_texts.t()
        sparse_t2i_scores = sparse_i2t_scores.t()
        with torch.no_grad():
            scores_dense_i2t = dense_imgs @ dense_texts.t()
            prob_dense_i2t = torch.softmax(
                scores_dense_i2t/self.temp, dim=1)
            prob_dense_t2i = torch.softmax(
                scores_dense_i2t.t()/self.temp, dim=1)
        loss = (self.ce(sparse_i2t_scores, prob_dense_i2t) +
                self.ce(sparse_t2i_scores, prob_dense_t2i))/2
        reg = (self.q_regularizer(sparse_texts) +
               self.d_regularizer(sparse_imgs))/2
        self.q_regularizer.step()
        self.d_regularizer.step()
        return loss, reg
