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
    
    def calculate_loss(self, predictions, labels):
        self.loss = np.mean(np.square(predictions - labels))
        return self.loss
    
    def calculate_gradient(self, predictions, labels):
        self.gradient = 2 * (predictions - labels) / labels.size
        return self.gradient

    


#BATCH
# O SHAPE DO BATCH TEM QUE SER (O TAMANHO DO BATCH, NUMERO DE INPUTS)


batch_size = 10

inputs = []
labels = []

layerInputs = Layer((11, 5))
layerHidden1 = Layer((5, 5))
layerHidden2 = Layer((5, 1))

csv = pd.read_csv("winequality-red.csv")
 
window_features = []
window_label = []


# PREPARA OS DADOS
for index, line in csv.iterrows():

    features = line[:-1].tolist()
    label = line[-1]

    window_features.append(features)
    window_label.append(label)
    
    # Se a janela estiver com mais de 10 linhas, remove a linha mais antiga
    if len(window_features) % batch_size == 0:
        inputs.append(np.array(window_features))
        labels.append(np.array(window_label))
        window_features.clear()
        window_label.clear()




#TREINA
for i, batch in enumerate(inputs):
    layerInputs.forward(batch)
    layerHidden1.forward(layerInputs.output)
    layerHidden2.forward(layerHidden1.output)
    print("##############")
    predictions = layerHidden2.output
    

    #calcula o erro
    loss = layerHidden2.calculate_loss(predictions, labels[i])

    

    print(loss)





