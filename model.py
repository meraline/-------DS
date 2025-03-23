"""
model.py - Определение модели RWKV для покерных данных
"""
import torch
import torch.nn as nn

# ---------------------- Модель RWKV ----------------------
class RWKV_Block(nn.Module):
    def __init__(self, hidden_dim):
        super(RWKV_Block, self).__init__()
        self.hidden_dim = hidden_dim

        # Параметры time-mixing
        self.time_decay = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Слои
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Состояние
        self.register_buffer("state", None, persistent=False)

    def reset_state(self):
        self.state = None

    def _init_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(self.time_decay.device)

    def forward(self, x):
        # x имеет форму [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.size()

        # Инициализация состояния
        if self.state is None or self.state.size(0) != batch_size:
            self.state = self._init_state(batch_size)

        output = []
        for t in range(seq_len):
            # Текущий временной шаг
            xt = x[:, t]

            # Time-mixing
            k = self.key(xt * self.time_mix_k + self.state * (1 - self.time_mix_k))
            v = self.value(xt * self.time_mix_v + self.state * (1 - self.time_mix_v))
            r = torch.sigmoid(self.receptance(xt * self.time_mix_r + self.state * (1 - self.time_mix_r)))

            # Обновление состояния
            self.state = xt + self.state * torch.exp(-torch.exp(self.time_decay))

            # Вычисление выхода
            out = r * self.output(v)
            output.append(out)

        return torch.stack(output, dim=1)

class SimpleRWKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(SimpleRWKV, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.rwkv_layers = nn.ModuleList([RWKV_Block(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def reset_states(self):
        for layer in self.rwkv_layers:
            layer.reset_state()

    def forward(self, x):
        # x имеет форму [batch_size, input_dim]
        x = self.embedding(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        for layer in self.rwkv_layers:
            residual = x
            x = layer(x)
            x = residual + x  # Остаточное соединение
            x = self.norm(x)
            x = self.dropout(x)

        x = x.squeeze(1)  # [batch_size, hidden_dim]
        return self.classifier(x)