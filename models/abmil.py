import torch
import torch.nn as nn
import torch.nn.functional as F


"""
https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
"""
class ABMIL(nn.Module):
    def __init__(self,dim_in=2048,dim_hid=256,num_classes=2,use_proj=False,use_gate=False,dropout=0):
        super(ABMIL, self).__init__()
        self.dim_in = dim_in # Encoded dim
        self.dim_hid = dim_hid # Hidden dim
        self.num_classes = num_classes
        self.use_gate = use_gate
        self.use_proj = use_proj
        self.dropout = dropout

        # feature projection
        self.proj = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in),
            nn.ReLU()
        )

        # gated attention (hyperbolic + sigmoid)
        if self.use_gate==True:
            self.attention_V = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_hid),
                nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_hid),
                nn.Sigmoid()
            )

            self.attention_weights = nn.Linear(self.dim_hid, 1)

        # attention (hyperbolic tangent)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_hid),
                nn.Tanh(),
                nn.Linear(self.dim_hid, 1)
            )


        self.classifier = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hid),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_hid, self.dim_hid),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_hid, self.num_classes),
        )

    def forward(self, x ):
        x = x.view(-1,self.dim_in)

        H = self.proj(x)  # NxL
        if self.use_gate==True:
            A_V = self.attention_V(H)  # NxD
            A_U = self.attention_U(H)  # NxD
            A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        else:
            A = self.attention(H)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, H)  # KxL

        Y_logit = self.classifier(M)
        return {
            'logit':Y_logit,
            'feature':M,
            'attention': A
        }