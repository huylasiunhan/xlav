import torch
from torch import nn
from src.config import Lab3Config

class BaseClassifier(nn.Module):
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    @torch.no_grad
    def predict_proba(self , x) -> torch.Tensor:
        result = self.forward(x)
        return torch.softmax(result , dim=-1)

    @torch.no_grad
    def predict(self, x):
        result = self.forward(x)
        return torch.argmax(result , dim=-1)
    
    def count_parameters(self) -> int:
        sum = 0
        for p in self.parameters():
            if p.requires_grad == True:
               sum += p.numel()
        return sum 

class MultiClassImageMLP(BaseClassifier):
    def __init__(self, input_dim, num_classes, hidden_dims = (512,256,128), dropout=0.25):
        super().__init__()
        if input_dim < 0 or num_classes <= 2 or hidden_dims is None:
            raise ValueError("Invalid input")
        
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def forward(self , x):
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)

class CurrencyCNN(BaseClassifier):
    @staticmethod
    def conv2d(c_in, c_out, kernel_size = 3, padding = 1, bias = False):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = kernel_size, bias = bias, padding = padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def __init__(self, in_channels , num_classes, dropout=0.25):
        super().__init__()
        if in_channels not in (1, 3) or num_classes < 2 or dropout > 1.0 or dropout < 0.0:
            raise ValueError("Invalid value")
        self.features = nn.Sequential(
            self.conv2d(in_channels, 32),
            self.conv2d(32, 64),
            self.conv2d(64, 128),
            self.conv2d(128, 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Invalid input")
        return self.features(x)

def build_model(cfg: Lab3Config):
    if cfg.model_type == "mlp":
        return MultiClassImageMLP(cfg.input_dim, cfg.num_classes, cfg.hidden_dims, cfg.dropout)
    elif cfg.model_type == "cnn":
        return CurrencyCNN(cfg.num_channels, cfg.num_classes, cfg.dropout)
    else:
        raise ValueError("Invalid model type")