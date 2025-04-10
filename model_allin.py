
"""
Модель RWKV для предсказания all-in решений
"""
import torch
import torch.nn as nn
from model import SimpleRWKV

class RWKVAllInModel(SimpleRWKV):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        # Для all-in у нас бинарная классификация (0 или 1)
        super().__init__(input_dim, hidden_dim, num_classes=2, num_layers=num_layers)
        
        # Добавляем дополнительный слой для улучшения обучения
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        x = super().forward(x)
        return self.final_activation(x)
