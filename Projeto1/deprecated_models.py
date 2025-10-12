from torch import nn

class LinearModel(nn.Module):
    def __init__(self, in_features=8, out_features=1):
        """
        in_features: Número de features que o dataset possui
        out_features: Classificação binária
        """
        super(LinearModel, self).__init__() # instancia o nn.Module
        self.l1 = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Passa os dados para a camada linear
        # v = Σ(wx) + b
        x = self.l1(x)
        # 2. Função do campo local induzido (não-linear)
        # y = φ(v)
        x = self.sigmoid(x)
        
        return x