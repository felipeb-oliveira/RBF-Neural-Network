import math
import random
import KMC

LEARNING_RATE = 0.3
MIN_ERROR_VAR = 0.5
SIGMA = 40

#calculo do modulo da distancia de dois vetores
def distanceModule(inputs1, inputs2):
    distance = []
    for i in range(0, len(inputs1)):
        distance.append(inputs1[i] - inputs2[i])
    
    sum = 0
    for i in range(0, len(inputs1)):
        sum += distance[i]**2
        
    sum = math.sqrt(sum)
    return sum

#função logistica
def logisticFunction(value):
    return 1/(1 + math.e ** (-value))

#derivada da função logística
def derLogisticFunction(value):
    return (logisticFunction(value) * (1 - logisticFunction(value)))

#função gaussiana de base radial
def radialBasisFunction(inputs, RBFCenter):
    
    sum = distanceModule(inputs, RBFCenter)
    
    return math.e ** -(((sum) ** 2) / SIGMA)
    

class Neuron:
    def __init__(self, RBF: bool, RBFCenter = 0):
        #variáveis do neurônio
        self.rawValue = 0
        self.outValue = 0
        self.RBFCenter = RBFCenter
        
        #variáveis de controle
        self.RBF = RBF
      
    def feed(self, inputs):
        sum = 0
        for x in inputs:
            sum += x

        self.rawValue = sum
        
        #caso seja RBF, realiza a função de base radial
        #se não, aplica-se a soma das entradas na função logística
        if(self.RBF):
            self.outValue = radialBasisFunction(inputs, self.RBFCenter)
            return self.outValue
        else:
            self.outValue =  logisticFunction(sum)
            return self.outValue
        
    
class Layer:
    def __init__(self, numNeurons, numInputs, RBF: bool, RBFCenters = []):
        #variáveis de controle
        self.numNeurons = numNeurons
        self.numInputs = numInputs
        self.RBF = RBF
        
        #variáveis da camada
        self.RBFCenters = RBFCenters
        self.neurons = []
        self.weights = []
        self.outputs = []
        self.inputs = []
        self.inputsNoBias = []
        self.deltaWeights = []
        self.error = []
        
        #se for RBF, cria-se uma camada com neuronios RBF sem bias anterior
        #se não, cria-se com neuronios normais com bias da camada anterior
        if(RBF):
            self.RBFOutputs = []
            
            for j in range(0, numInputs):
                self.inputs.append(0)
                self.inputsNoBias.append(0)
            
            for i in range(0, numNeurons):
                self.neurons.append(Neuron(True, RBFCenters[i]))
                self.error.append(1)
                self.outputs.append(0)
                self.weights.append([])
                self.deltaWeights.append([])
                self.RBFOutputs.append([])
    
                for j in range(0, numInputs):
                    self.weights[i].append(1)
                    self.deltaWeights[i].append(0)
            
        else:
            #+1 referente ao bias
            for j in range(0, numInputs + 1):
                self.inputs.append(0)
                
            for i in range(0, numNeurons):
                self.neurons.append(Neuron(False))
                self.error.append(1)
                self.outputs.append(0)
                self.weights.append([])
                self.deltaWeights.append([])
                    
                #+1 referente ao bias
                for j in range(0, numInputs + 1):
                    self.weights[i].append(random.random())
                    self.deltaWeights[i].append(0)
        
        
    def feedForward(self, inputs):

        self.inputsNoBias = inputs
        
        #adiciona o input do bias caso não seja RBF
        if(not self.RBF):
            inputs = [1] + inputs
        
        
        for i in range(len(inputs)):
            self.inputs[i] = inputs[i]
        
        for i in range(0, len(self.neurons)):
            #multiplica o input pelo peso da ligação correspondente
            for j in range(0, len(inputs)):
                inputs[j] = self.inputs[j] * self.weights[i][j] 
                
            #o objeto neuronio realiza o calculo
            self.outputs[i] = self.neurons[i].feed(inputs)

                
        return self.outputs
    
    
    def backPropagation(self, inputs, desiredOutputs):
        #feedforward inicial para ter referencia para o erro
        self.feedForward(inputs)

        for i in range(0, len(desiredOutputs)):
            #calculo do erro
            self.error[i] = desiredOutputs[i] - self.outputs[i]
            
            #calculo da variação de pesos (baseado na regra Delta)
            for j in range(0, len(self.inputs)):
                self.deltaWeights[i][j] = self.error[i] * self.inputs[j] * LEARNING_RATE * derLogisticFunction(self.outputs[i])
            
        
    def updateWeights(self):
        
        #soma a matriz de variação de peso à matriz peso
        for i in range(0, len(self.outputs)):
            for j in range(0, len(self.inputs)):
                self.weights[i][j] += self.deltaWeights[i][j]
                

    
class RBFNet:
    def __init__(self, neuronsPerLayer, dataset):
        
        #variaveis de controle
        self.numLayers = len(neuronsPerLayer) - 1
        self.numInputs = neuronsPerLayer[0]
        self.numOutputs = neuronsPerLayer[self.numLayers]
        self.numHiddenLayers = self.numLayers - 1
        self.neuronsPerLayer = neuronsPerLayer
        
        #variaveis da rede
        self.layers = []
        self.inputs = []
        self.outputs = []
        self.RBFCenters = []

        #inicialização de veriáveis
        for i in range(0, self.numLayers - 1):
            self.RBFCenters.append([])
        
        #calculo dos centros RBF para cada camada RBF (no caso só se usa uma)
#        centers = self.findRBFCenters(dataset)
#        self.RBFCenters = centers
               
        for i in range(1, self.numLayers):
            self.layers.append(Layer(neuronsPerLayer[i], neuronsPerLayer[i-1], False, self.RBFCenters[i-1]))

        
        self.layers.append(Layer(self.numOutputs, neuronsPerLayer[self.numLayers - 1], False))

        
    def feedForward(self, inputs):
        outputs = inputs
        
        #realiza o feedforward em cada camada
        for l in self.layers:    
            outputs = l.feedForward(outputs)
            
        self.outputs = outputs
        
        return outputs
    
    def getError(self, desiredOutputs):
        sum = 0

        #calculo da energia do erro da saída da rede
        for i in range(0, self.numOutputs):
            sum += ((desiredOutputs[i] - self.outputs[i]) ** 2) / 2
            
        return sum
    
    def updateNet(self):
        
        #atualiza os pesos de cada camada
        for l in self.layers:
            if(not l.RBF):
                l.updateWeights()
            
    
    def findRBFCenters(self, dataset):
        
        #utiliza o algoritmo k means clustering para encontrar os centros RBF
        dimensions = len(dataset[0])
        centers = []
        
        for i in range(0, self.numLayers - 1):
            centers.append(KMC.KMC(dataset, self.neuronsPerLayer[i+ 1], dimensions))
            
        return centers
            

    def backPropagation(self, inputs, desiredOutputs):
        
        #realiza o feedforward para referencia de erro
        self.feedForward(inputs)

        #realiza o backpropagation em cada camada, da ultima ate a primeira
        for i in range(self.numLayers - 1, -1, -1):
            if(not self.layers[i].RBF):
                self.layers[i].backPropagation(self.layers[i].inputsNoBias, desiredOutputs)
                
    def getResult(self, inputs):
        #encontra-se o output para o dado input
        result = self.feedForward(inputs)
           
        #arredonda o output para 1 ou 0, sendo maior ou menor que 0,5
        for i in range(0, len(result)):
            if(result[i] < 0.5):
                result[i] = 0
            else:
                result[i] = 1
            
        return result
        
        
    def train(self, datasetInputs, datasetOutputs):

        #realiza o primeiro feedforward para ter referencia de erro anterior
        lastTotalError = 0
        for i in range(0, len(datasetInputs)):
            self.feedForward(datasetInputs[i])
            lastTotalError += self.getError(datasetOutputs[i])
            
        print("Total Error: " + "%.3f" % lastTotalError)
        
        #enquanto a variaçao de erro for menor que a minima, continua o loop
        while(True):
            totalError = 0
            
            #backpropagation, atualização de pesos da rede e soma do erro de cada dado do dataset
            for i in range(0, len(datasetInputs)):
                self.backPropagation(datasetInputs[i], datasetOutputs[i])
                totalError += self.getError(datasetOutputs[i])
                self.updateNet()
                
            #print do erro da era
            print("Total Error: " + "%.3f" % totalError + " | Error var = " + "%.3f" % (lastTotalError - totalError))
    
            #break caso a variaçao seja menor
            if(lastTotalError - totalError <= MIN_ERROR_VAR):
                break
            
            #atualizaçao do erro anterior
            lastTotalError = totalError
            
        return totalError
                
            
        
            
               
        
        
            
        
    
    
    
    

