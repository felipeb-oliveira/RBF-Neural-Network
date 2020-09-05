import RBFNet
import random

file = open("semeion.data", "r+")
fileLines = file.readlines()
numData = len(fileLines)

datasetOutputs = []
datasetInputs = []

numOutputs = 10

for i in range(0, numData):
    fileLines[i] = fileLines[i].split(' ')
    fileLines[i].pop()
    
    datasetOutputs.append([])
    datasetInputs.append([])
    
    for j in range(0, len(fileLines[i]) - numOutputs):
        datasetInputs[i].append(float(fileLines[i][j]))
    
    for j in range(len(fileLines[i]) - numOutputs, len(fileLines[i])):
        datasetOutputs[i].append(float(fileLines[i][j]))
        
numInputs = len(fileLines[0]) - numOutputs

testPercentage = 0.3

trainInputs = datasetInputs
trainOutputs = datasetOutputs
testInputs = []
testOutputs = []

numTest = int(testPercentage * numData)
numTrain = numData - numTest

randRange = numData - 1

for i in range(0, numTest):
    rand = random.randint(0, randRange)
    testInputs.append(trainInputs.pop(rand))
    testOutputs.append(trainOutputs.pop(rand))
    randRange -= 1
    
    
print("Creating net...")    
net = RBFNet.RBFNet([numInputs, int(1.2*numInputs),numOutputs], trainInputs)

print("Net created!")

print("Starting training...")
finalError = net.train(trainInputs, trainOutputs)
print("Training finished!")
print("Final error: " + str(finalError))

print("")

print("Starting test...")

hit = 0
for i in range(0, numTest):
    if(net.getResult(testInputs[i]) == testOutputs[i]):
        hit += 1

hitPercentage = hit/numTest

print("Testing finished!")
print("Final hit percentage: " + str(hitPercentage))
    
file.close()


        
    



