import pandas as pd
import numpy as np
import math

np.random.seed(1)

#gera uma matrix do mesmo formato da entrada


#Classe
class Layer:
    def __init__(self, shape):
        self.shape = shape
        # matrix de pesos
        self.weights = 0.1 * np.random.randn(shape[0], shape[1])
        self.bias = np.zeros((1,shape[1]))
        
        

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def activate_reLU(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


#BATCH
# O SHAPE DO BATCH TEM QUE SER (O TAMANHO DO BATCH, NUMERO DE INPUTS)


batch_size = 10

inputs = []

    

layerInputs = Layer((11, 5))
layerHidden1 = Layer((5, 5))
layerHidden2 = Layer((5, 1))

features = pd.read_csv("winequality-red.csv")
features = features.iloc[:, :-1]

labels = pd.read_csv("winequality-red.csv")
labels = labels.iloc[:, -1]

matrizes = [] 
window = []


for index, row in features.iterrows():
    # Adiciona a linha atual à janela deslizante
    window.append(row.values)
    
    # Se a janela estiver com mais de 10 linhas, remove a linha mais antiga
    if len(window) > batch_size:
        window.pop(0)
    
    # Se a janela tiver 10 linhas, cria uma matriz e adiciona à lista de matrizes
    if len(window) == batch_size:
        matriz = np.array(window)
        matrizes.append(matriz)


print(matrizes)
print(labels)

