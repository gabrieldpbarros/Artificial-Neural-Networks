import matplotlib.pyplot as plt
from typing import Dict

def plotLosses(loss1: Dict[int, float], loss2: Dict[int ,float], loss1_label: str, loss2_label: str) -> None:
    plt.close("all")
    plt.figure()
    plt.plot(loss1.keys(), loss1.values(), label=loss1_label)
    plt.plot(loss2.keys(), loss2.values(), label=loss2_label)
    plt.title(f"{loss1_label} e {loss2_label}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()
    plt.show()