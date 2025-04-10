
"""
Модель RWKV для предсказания размеров ставок
"""
import torch
import torch.nn as nn
from model import SimpleRWKV

class RWKVSizeModel(SimpleRWKV):
    def __init__(self, input_dim, hidden_dim, num_size_categories, num_layers=2):
        super().__init__(input_dim, hidden_dim, num_size_categories, num_layers)
        
    def forward(self, x):
        # Используем базовую логику RWKV
        return super().forward(x)
