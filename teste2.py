#vamo fazer isso do meu jeito 
import numpy as np 
import pandas as pd


#criando a classe do neuronio
class RedeNeural: 
    def __init__(self, formatoRedeNeural):
        #array com o formato da rede neural (numero de neuronios em cada camdada)
        self.formatoRedeNeural = formatoRedeNeural
        #o formato dos pesos deve ser um array tridimensional onde (sessoes de pesos, n-input-neurons, n-output-neurons) 
        self.weights = []
        self.bias = []
        #o formato dos inputs deve ser um array bidimensional onde (tamanho do batch, n-input-neurons)
        self.inputs = np.array([])


    def setInputs(self, inputs):
        #inputs é um batch de entradas (tamanho do batch x numero de entradas)
        #verifica se o shape das entradas é o mesmo do formato da rede neural
        if inputs.shape[1] != self.formatoRedeNeural[0]:
            raise ValueError("O numero de entradas não é o mesmo do formato da rede neural")
        self.inputs = inputs

    def setWeights(self):
        #TERCEIRO NIVEL 
        #o numero de sessões será igual ao numero de camadas menos 1
        num_sessions = len(self.formatoRedeNeural) - 1
        num_layers = num_sessions + 1

        #define os pesos
        for x in range(num_sessions):
            # o formato dos pesos é (n-input-neurons, n-output-neurons)
            self.weights.append(np.random.randn(self.formatoRedeNeural[x], self.formatoRedeNeural[x+1]))
        
        #define os bias
        for x in range(num_layers):
            if x == 0:
                continue
            self.bias.append(np.ones(self.formatoRedeNeural[x]))
        
        
        #self.weights = np.array(self.weights)

    def forward(self):
        #para cada layer, exceto a primeira
        for x in range(1, len(self.formatoRedeNeural)):
            #multiplica os pesos por os inputs
            print(f"########## OUTPUT LAYER: {x+1} ############")

            #soma ponderada
            self.inputs = np.dot(self.inputs, self.weights[x-1])
            for y in self.inputs:
                y += self.bias[x-1]
            
            #self.inputs += self.bias[x-1]
            self.inputs = self.activate_relu(self.inputs)
            print(self.inputs)
            

            print(f"########## OUTPUT LAYER: {x+1} ############\n\n")

            
    def calculate_error(self, label): 
        #calcula o erro medio quadrático
        n = len(label)
        #mse = ((label - self.inputs) ** 2).sum() / n
        mse = np.mean(np.square(label - self.inputs))
        return mse
    
    def derivative(self, label):
        return 2 * (self.inputs - label)


    def activate_relu(self, inputs):
        return np.maximum(0, inputs)
    
    def update_weights(self, learning_rate, derivative):
        for x in self.weights:
            if(derivative.any()):
                x -= learning_rate
            else:
                x += learning_rate


if __name__ == "__main__":
    treino = np.array([[6, 9], [4,2], [4,5], [8,1], [1,9], [5,5], [2,6], [2,8], [9,5], [6,4], [3,3], [9,8]])
    #treino = np.array([5,8])
    label = np.array([69])
    label = label.reshape(1,1)
    learning_rate = 0.01


    #divide o array em pedaços
    #treino = np.array_split(treino, 6)
    
    
    
    rede = RedeNeural((2, 1))
    
 
    
    for x in range(100000):
        rede.setInputs(treino)
        rede.setWeights()
        rede.forward()

        print("############# ERRO MÉDIO QUADRATICO #############")
        mse = rede.calculate_error(label)
        print(mse)

        derivada = rede.derivative(label)
        rede.update_weights(learning_rate, derivada)


    

    
