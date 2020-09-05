import math
import random

class Cluster:
    def __init__(self, position):
        self.position = position
        
    def __str__(self):
        return self.position
        
    def getPosition(self):
        return self.position
    
    def setPosition(self, position):
        self.position = position



class Point:
    def __init__(self, position):
        self.position = position
        
    def __str__(self):
        return self.position
    
    def getPosition(self):
        return self.position
    
    #assignedCluster se refere ao cluster mais proximo do dado no momento
    def assignCluster(self, clusterNumber):
        self.assignedCluster = clusterNumber
        
    def getAssignedCluster(self):
        return self.assignedCluster
        
        
        
#distancia euclidiana entre dois pontos
def distance(point1, point2):
    distanceVector = []
    for i in range(0, len(point1) - 1):
        distanceVector.append(point1[i] - point2[i])
        
    result = 0
    for i in range(0, len(distanceVector) - 1):
        result += distanceVector[i]**2
    
    result = math.sqrt(result)
    return result
         
    
#procura o cluster mais proximo de um ponto
def closestCluster(clusters, position):
    closestCluster = 0
    clusterIndex = 0
    
    for c in clusters:
        if distance(position, c.getPosition()) < distance(position, clusters[closestCluster].getPosition()):
            closestCluster = clusterIndex
            
        clusterIndex += 1
            
    return closestCluster


#encontra a posição média em um dataset de pontos
def meanPosition(points, dimensions):
    result = []
                     
    for i in range(0, dimensions):
        sum = 0
        for p in points:
            sum += p.getPosition()[i]
            
        sum = sum/len(points)
        result.append(sum)
        
    return result


#retorna os pontos no agrupamento de um centro
def clusterPoints(numCluster, points):
    result = []
    
    for p in points:
        if p.getAssignedCluster() == numCluster:
            result.append(p)
            
    return result

#converte um dataset para uma lista de pontos
def setToPoints(dataset):
    points = []
    
    for s in dataset:
        points.append(Point(s))
        
    return points
  
        

def KMC(dataset, nClusters, dimensions):
    
    clusters = []
    
    #converte os dados do dataset em classes de pontos
    dataset = setToPoints(dataset)
    
    #gera posição aleatoria para os k centros e coloca na variável clusters
    for i in range(0, nClusters):
        randomPosition = []
        
        for j in range(0, dimensions):
            randomPosition.append(random.random())
            
        clusters.append(Cluster(randomPosition))
        
    #realiza os passos descritos no relatório até convergencia
    convergenceCheck = 0
    while(convergenceCheck == 0):
        for p in dataset:
            p.assignCluster(closestCluster(clusters, p.getPosition()))
            
        convergenceCheck = 0
        clusterIndex = 0
        
        for c in clusters:
            
            if clusterPoints(clusterIndex, dataset) != []:
                newPosition = meanPosition(clusterPoints(clusterIndex, dataset), dimensions)
                if c.getPosition != newPosition:
                    convergenceCheck = 1
                
                c.setPosition(newPosition)
            
            clusterIndex += 1
    
    #converte os clusteres para uma lista
    clustersCenters = []
    for c in clusters:
        clustersCenters.append(c.getPosition())

        
    return clustersCenters

        