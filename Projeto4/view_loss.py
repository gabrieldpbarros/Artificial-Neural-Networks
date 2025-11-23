import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

PATH = "assets/"

def plotLosses(loss1: Dict[int, float],
               loss2: Dict[int ,float],
               loss1_label: str,
               loss2_label: str,
               saving: str) -> None:
    plt.close("all")
    plt.figure()
    plt.plot(loss1.keys(), loss1.values(), label=loss1_label)
    plt.plot(loss2.keys(), loss2.values(), label=loss2_label)
    plt.title(f"{loss1_label} e {loss2_label}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()
    plt.savefig(os.path.join(PATH, saving))
    plt.show()

def plotSeries(real_values: np.ndarray,
               predicted_values: np.ndarray,
               real_label: str,
               predicted_label: str,
               saving: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(real_values, label=real_label)
    plt.plot(predicted_values, label=predicted_label)
    plt.title("Série temporal - Valores reais vs previstos")
    plt.xlabel("Tempo")
    plt.ylabel("Average Players")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PATH, saving))
    plt.show()