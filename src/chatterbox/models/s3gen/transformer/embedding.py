# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.pe = None  # Will be initialized lazily

    def _generate_pe(self, length: int, device: torch.device, dtype: torch.dtype):
        position = torch.arange(0, length, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=dtype, device=device) *
                             -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(length, self.d_model, dtype=dtype, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pe is None or self.pe.size(1) < x.size(1) + offset:
            self._generate_pe(x.size(1) + offset, x.device, x.dtype)
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        if self.pe is None or self.pe.size(1) < offset + size:
            self._generate_pe(offset + size, torch.device('cpu'), torch.float32)
        return self.pe[:, offset:offset + size]

