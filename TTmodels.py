import math
import torch
import torch.nn as nn


# Helpers
def insert_gap_zeros(x: torch.Tensor, block: int = 500, gap: int = 2) -> torch.Tensor:
  
    had_channel = (x.dim() == 3)
    if had_channel:
        x = x.squeeze(1)

    B, L = x.shape
    n_blocks = L // block
    out_len = L + (n_blocks - 1) * gap
    out = x.new_zeros(B, out_len)

    step = block + gap
    for i in range(n_blocks):
        src0 = i * block
        dst0 = i * step
        out[:, dst0:dst0 + block] = x[:, src0:src0 + block]

    return out.unsqueeze(1) if had_channel else out


#########################################
# TT-MLP (probit + product fusion)
class BinaryFertilityModel(nn.Module):
    """
    a, b tower liabilities
    log_p log of product (of probits)
    """
    def __init__(self, input_dim, dropout_rate: float = 0.0, size: int = 128, clamp_scores: float = 8.0):
        super().__init__()
        self.sire_net = self._make_tower(input_dim, dropout_rate, size)
        self.dam_net  = self._make_tower(input_dim, dropout_rate, size)

        self.alpha_s = nn.Parameter(torch.tensor(1.0))
        self.alpha_d = nn.Parameter(torch.tensor(1.0))
        self.beta_s  = nn.Parameter(torch.tensor(0.0))
        self.beta_d  = nn.Parameter(torch.tensor(0.0))

        self.clamp_scores = clamp_scores

    @staticmethod
    def _make_tower(input_dim, dr=0.0, hidden=128):
        return nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sire_input, dam_input):
        a = self.sire_net(sire_input).squeeze(-1)
        b = self.dam_net(dam_input).squeeze(-1)

        a = self.alpha_s * a + self.beta_s
        b = self.alpha_d * b + self.beta_d

        if self.clamp_scores is not None:
            a = a.clamp(-self.clamp_scores, self.clamp_scores)
            b = b.clamp(-self.clamp_scores, self.clamp_scores)

        log_p = torch.special.log_ndtr(a) + torch.special.log_ndtr(b)
        return a, b, log_p


##########################################
# Conv Two-Tower (probit + product fusion)
class BinaryFertilityModelConv(nn.Module):
    """
    a, b are tower liabilities
    log_p log of product (of probits)
    """
    def __init__(self, input_channels, conv: int = 1, dropout_rate: float = 0.0,
                 pool: int = 1024, size: int = 128, clamp_scores: float = 8.0):
        super().__init__()
        self.sire_net = self._make_tower(input_channels, conv, dropout_rate, pool, size)
        self.dam_net  = self._make_tower(input_channels, conv, dropout_rate, pool, size)

        self.alpha_s = nn.Parameter(torch.tensor(1.0))
        self.alpha_d = nn.Parameter(torch.tensor(1.0))
        self.beta_s  = nn.Parameter(torch.tensor(0.0))
        self.beta_d  = nn.Parameter(torch.tensor(0.0))

        self.clamp_scores = clamp_scores

    @staticmethod
    def _make_tower(input_channels, nc=1, dr=0.0, ps=1024, hidden=128):
        return nn.Sequential(
            nn.Conv1d(1, nc, kernel_size=3, padding=1),
            nn.BatchNorm1d(nc),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(ps),
            nn.Flatten(),
            nn.Linear(ps * nc, hidden), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sire_input, dam_input):
        sire_input = insert_gap_zeros(sire_input, block=500, gap=2)
        dam_input  = insert_gap_zeros(dam_input,  block=500, gap=2)

        # shape is (B, 1, L)
        if sire_input.dim() == 2:
            sire_input = sire_input.unsqueeze(1)
        elif sire_input.dim() == 3 and sire_input.size(1) != 1 and sire_input.size(2) == 1:
            sire_input = sire_input.transpose(1, 2)
        elif sire_input.dim() != 3:
            raise RuntimeError(f"Unexpected sire_input shape: {tuple(sire_input.shape)}")

        if dam_input.dim() == 2:
            dam_input = dam_input.unsqueeze(1)
        elif dam_input.dim() == 3 and dam_input.size(1) != 1 and dam_input.size(2) == 1:
            dam_input = dam_input.transpose(1, 2)
        elif dam_input.dim() != 3:
            raise RuntimeError(f"Unexpected dam_input shape: {tuple(dam_input.shape)}")

        a = self.sire_net(sire_input).squeeze(-1)
        b = self.dam_net(dam_input).squeeze(-1)

        a = self.alpha_s * a + self.beta_s
        b = self.alpha_d * b + self.beta_d

        if self.clamp_scores is not None:
            a = a.clamp(-self.clamp_scores, self.clamp_scores)
            b = b.clamp(-self.clamp_scores, self.clamp_scores)

        log_p = torch.special.log_ndtr(a) + torch.special.log_ndtr(b)
        return a, b, log_p


###########################################
# LASSO Two-Tower (probit + product fusion)
class BinaryFertilityLassoTwoTower(nn.Module):
    """
    a, b are tower liabilities
    log_p log of product (of probits)
    """
    def __init__(self, input_dim, clamp_scores: float = 8.0):
        super().__init__()
        self.sire = nn.Linear(input_dim, 1, bias=True)
        self.dam  = nn.Linear(input_dim, 1, bias=True)

        self.alpha_s = nn.Parameter(torch.tensor(1.0))
        self.alpha_d = nn.Parameter(torch.tensor(1.0))
        self.beta_s  = nn.Parameter(torch.tensor(0.0))
        self.beta_d  = nn.Parameter(torch.tensor(0.0))

        self.clamp_scores = clamp_scores

        for lin in (self.sire, self.dam):
            nn.init.normal_(lin.weight, mean=0.0, std=0.01)
            nn.init.zeros_(lin.bias)

    def forward(self, sire_input, dam_input):
        a = self.sire(sire_input).squeeze(-1)
        b = self.dam(dam_input).squeeze(-1)

        a = self.alpha_s * a + self.beta_s
        b = self.alpha_d * b + self.beta_d

        if self.clamp_scores is not None:
            a = a.clamp(-self.clamp_scores, self.clamp_scores)
            b = b.clamp(-self.clamp_scores, self.clamp_scores)

        log_p = torch.special.log_ndtr(a) + torch.special.log_ndtr(b)
        return a, b, log_p

    @torch.no_grad()
    def proximal_step(self, lr, l1_lambda, include_bias: bool = False):
        def soft_thresh(p, t):
            return torch.sign(p) * torch.clamp(p.abs() - t, min=0.0)

        thresh = float(lr) * float(l1_lambda)
        self.sire.weight.copy_(soft_thresh(self.sire.weight, thresh))
        self.dam.weight.copy_(soft_thresh(self.dam.weight, thresh))
        if include_bias:
            self.sire.bias.copy_(soft_thresh(self.sire.bias, thresh))
            self.dam.bias.copy_(soft_thresh(self.dam.bias, thresh))


#################################################################
# MLP taking concatented input to only predict agg outcome 4by128
class BinaryFertilityCompat(nn.Module):

    def __init__(
        self,
        input_dim,
        dropout_rate: float = 0.0,
        size: int = 256,
        clamp_scores: float = 8.0,
    ):
        super().__init__()

        print(128/size)

        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
        )

        self.clamp_scores = clamp_scores

    def forward(self, sire_input, dam_input):

        x = torch.cat([sire_input, dam_input], dim=1)

        logit = self.net(x).squeeze(-1)

        if self.clamp_scores is not None:
            logit = logit.clamp(
                -self.clamp_scores,
                self.clamp_scores,
            )

        log_p = torch.special.log_ndtr(logit)

        a = 0
        b = 0

        return a, b, log_p