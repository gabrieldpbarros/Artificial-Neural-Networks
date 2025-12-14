import os
import torch
import torch.optim as optim

from torch import nn
from models.simple_cnn import cnnModel
from utils.load_data import getDataLoaders
from utils.view_loss import plotLosses, plotCMatrix

EPOCHS = 30
DEVICE = "cuda"
PATH = "db/"

def mainPipeline(device, db_path):
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    train_loader, val_loader, test_loader = getDataLoaders(db_path)

    simpleModel = cnnModel(
        device=DEVICE,
        filters_list=[32, 64]
    )

    completeModel = cnnModel(
        device=DEVICE,
        filters_list=[32, 64, 128],
        dropout=True,
        batch_norm=True,
        GAP=True
    )

    optimizer_simple = optim.Adam(simpleModel.model.parameters(), lr=0.001)
    optimizer_complete = optim.Adam(completeModel.model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = EPOCHS

    best_simple_val_loss = float('inf')
    best_complete_val_loss = float('inf')
    for epoch in range(num_epochs):
        simpleModel.trainModel(train_loader, loss_fn, optimizer_simple)
        completeModel.trainModel(train_loader, loss_fn, optimizer_complete)

        simpleModel.validateModel(val_loader, loss_fn)
        completeModel.validateModel(val_loader, loss_fn)

        if simpleModel.val_loss[-1] < best_simple_val_loss:
            best_simple_val_loss = simpleModel.val_loss[-1]
            simpleModel.saveCeckpoint("models/best_simple_cnn_model.pth")
            print(f"Epoch {epoch+1}: Melhor modelo (simples) salvo! (Val Loss: {best_simple_val_loss:.4f})")

        if completeModel.val_loss[-1] < best_complete_val_loss:
            best_complete_val_loss = completeModel.val_loss[-1]
            completeModel.saveCheckpoint("models/best_complete_cnn_model.pth")
            print(f"Epoch {epoch+1}: Melhor modelo (completo) salvo! (Val Loss: {best_complete_val_loss:.4f})")

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Simple Train Loss: {simpleModel.train_loss[epoch]:.4f} | Simple Val Loss: {simpleModel.val_loss[epoch]:.4f}")
            print(f"Epoch {epoch+1}/{EPOCHS} | Complete Train Loss: {completeModel.train_loss[epoch]:.4f} | Complete Val Loss: {completeModel.val_loss[epoch]:.4f}")

    plotLosses(
        "Modelo Simples",
        simpleModel.train_loss,
        simpleModel.val_loss,
        "Erro de treino",
        "Erro de validação",
        EPOCHS,
        "assets/Simple_CNN_Losses.png"
    )

    plotLosses(
        "Modelo Completo",
        completeModel.train_loss,
        completeModel.val_loss,
        "Erro de treino",
        "Erro de validação",
        EPOCHS,
        "assets/Complete_CNN_Losses.png"
    )

    simpleModel.model.load_state_dict(torch.load("models/best_simple_cnn_model.pth"))
    simpleModel.testModel(test_loader)
    print(f"--- Relatório Final (Modelo Simples) ---")
    print(f"Acurácia (Accuracy):   {simpleModel.acc*100:.2f}%")
    print(f"Precisão (Precision):  {simpleModel.prec*100:.2f}%")
    print(f"Sensibilidade (Recall):{simpleModel.rec*100:.2f}%")
    print(f"F1 Score:              {simpleModel.f1*100:.2f}%")

    completeModel.model.load_state_dict(torch.load("models/best_complete_cnn_model.pth"))
    completeModel.testModel(test_loader)
    print(f"--- Relatório Final (Modelo Completo) ---")
    print(f"Acurácia (Accuracy):   {completeModel.acc*100:.2f}%")
    print(f"Precisão (Precision):  {completeModel.prec*100:.2f}%")
    print(f"Sensibilidade (Recall):{completeModel.rec*100:.2f}%")
    print(f"F1 Score:              {completeModel.f1*100:.2f}%")

    plotCMatrix(
        simpleModel.test_class,
        simpleModel.predicted_class,
        ['Normal', 'Pneumonia'],
        "assets/Simple_CNN_ConfusionMatrix.png"
    )

    plotCMatrix(
        completeModel.test_class,
        completeModel.predicted_class,
        ['Normal', 'Pneumonia'],
        "assets/Complete_CNN_ConfusionMatrix.png"
    )

if __name__ == "__main__":
    mainPipeline(DEVICE, PATH)