import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pylab import *
from FinalProjectNeuron import Neuron
import math
from sklearn.preprocessing import normalize

class GenreClassifier:
	def __init__(self):
		self.timeStep = 2
		self.learningRate = 0.001
		self.spikingThreshold = 0.8			#Found through testing my specific neurons
		self.inputLayerSize = 138
		self.numSeconds = 2005						#Number of input neurons/pixels
		self.timeThreshold = 10  				#Time to simulate neuron for

		self.classifications = 2
		self.hiddenLayerNum = 3
		self.neuronPerLayer = [10, 20, 10]

		self.dataList = []
		self.isFirst = 0

		self.inputLayer = []
		for i in range(self.inputLayerSize):
			self.inputLayer.append(Neuron(self.timeThreshold,0))

		self.middleLayer = []
		currNumInputs = 138
		for i in range(self.hiddenLayerNum):
			currLayer = []
			for j in range(self.neuronPerLayer[i]):
				currLayer.append(Neuron(self.timeThreshold, currNumInputs))
			self.middleLayer.append(currLayer)
			currNumInputs = self.neuronPerLayer[i]

		self.outputLayer = Neuron(self.timeThreshold, 0)	

		weights = []
		for i in range(math.floor(currNumInputs/2)):
			self.outputLayer.weights.append(math.ceil(uniform(0,1000))/1000)
		for i in range(math.floor(currNumInputs/2), currNumInputs):
			self.outputLayer.weights.append(math.ceil(uniform(0,1000))/1000)


	def getMetalFiles(self):
		self.metalFiles = []													
		for i in range(5):
			currName = "metal specs/metal.0000"
			currName = currName + str(i)
			currName += ".au.wav.csv"
			self.metalFiles.append(currName)

		# for i in range(10,30):
		# 	currName = "metal specs/metal.000"
		# 	currName += str(i)
		# 	currName += ".au.wav.csv"
		# 	self.metalFiles.append(currName)

		for j in range(5):
			array = np.genfromtxt(self.metalFiles[j], delimiter=',')
			array = np.transpose(array)
			array = array[0:250,:]

			labeledArray = []

			for i in range(array.shape[0]):
				labeledArray.append(np.append(array[i], 1))

			labeledArray = np.array(labeledArray)
			if(self.isFirst == 0):
				self.dataList = labeledArray
				self.dataList = np.array(self.dataList)
				self.isFirst = 1
			else:
				self.dataList = np.concatenate((self.dataList, labeledArray), axis=0)

	def getClassificationMetalInput(self):
		metalFiles = []		
		dataList = []
		for i in range(25,30):
			currName = "metal specs/metal.000"
			currName = currName + str(i)
			currName += ".au.wav.csv"
			metalFiles.append(currName)

		for j in range(5):
			array = np.genfromtxt(metalFiles[j], delimiter=',')
			array = np.transpose(array)
			array = array[0:100,:]

			labeledArray = []

			for i in range(array.shape[0]):
				labeledArray.append(np.append(array[i], 1))

			labeledArray = np.array(labeledArray)
			dataList.append(labeledArray)

		return dataList

	def getClassificationClassicalInput(self):
		classicalFiles = []		
		dataList = []
		for i in range(25,30):
			currName = "classical specs/classical.000"
			currName = currName + str(i)
			currName += ".au.wav.csv"
			classicalFiles.append(currName)

		for j in range(5):
			array = np.genfromtxt(classicalFiles[j], delimiter=',')
			array = np.transpose(array)
			array = array[0:100,:]

			labeledArray = []

			for i in range(array.shape[0]):
				labeledArray.append(np.append(array[i], 0))

			labeledArray = np.array(labeledArray)
			dataList.append(labeledArray)

		return dataList


	def getClassicalFiles(self):
		self.classicalFiles = []													
		for i in range(5):
			currName = "classical specs/classical.0000"
			currName += str(i)
			currName += ".au.wav.csv"
			self.classicalFiles.append(currName)

		# for i in range(10,30):
		# 	currName = "classical specs/classical.000"
		# 	currName += str(i)
		# 	currName += ".au.wav.csv"
		# 	self.classicalFiles.append(currName)

		for j in range(5):
			array = np.genfromtxt(self.classicalFiles[j], delimiter=',')
			array = np.transpose(array)
			array = array[0:250,:]

			labeledArray = []

			for i in range(array.shape[0]):
				labeledArray.append(np.append(array[i], 0))

			labeledArray = np.array(labeledArray)
			if(self.isFirst == 0):
				self.dataList = labeledArray
				self.dataList = np.array(self.dataList)
				self.isFirst = 1
			else:
				self.dataList = np.concatenate((self.dataList, labeledArray), axis=0)

	def train(self):
		input = self.dataList
		numFired = 0
		numNotFired = 0
		avgSpikeRate = 0
		for k in range(len(self.inputLayer)):
			neuron = self.inputLayer[k]
			currentSpikeRate = 0
			totalSpikeRate = 0
			numIncreased = 0
			numDecreased = 0
		
			currentSpikeRate = 0
			for i in range(len(input)):
				currentSpikeRate = neuron.runNeuron(input[i][k]*15+7.9)
				neuron.spikeRateForData.append(currentSpikeRate)
			for i in range(len(input)):
				if(currentSpikeRate >= self.spikingThreshold):					
					neuron.numFired += 1
					numIncreased+=1
				elif (currentSpikeRate < self.spikingThreshold):			
					neuron.notFired += 1
					numDecreased += 1

			if(neuron.numFired > neuron.notFired):
				# print("Fired! ")
				numFired+=1
				neuron.fired = 1
			else:
				# print("Not Fired for input: ", )
				numNotFired += 1							#Store current spike rate in array for training next
		# print(neuron.weights,"\nNum fired: ",numFired, " Num not fired:  ", numNotFired)
		# print("Average Spike Rate: ", avgSpikeRate, " ", avgSpikeRate/len(input))
		
		print("Training layer 1...\n\n")

		for k in range(len(self.middleLayer[0])):
			neuron = self.middleLayer[0][k]
			currentSpikeRate = 0
			totalSpikeRate = 0
			numFired = 0

			for i in range(len(input)):
				totalSpikeRate += currentSpikeRate
				# preSpikeRate = 
				currentSpikeRate = 0
				for j in range(len(input[0])-1):
					multiplier = 1
					if(input[i][138] == 0):
						multiplier *= 0.7
					currentSpikeRate += neuron.runNeuron(multiplier*(self.inputLayer[j].spikeRateForData[i])*neuron.weights[j]*2.2)
					neuron.spikeRateForData.append(currentSpikeRate)									#Store current spike rate in array for training next
				# print("Curr spike rate: ", currentSpikeRate)
				for j in range(len(input[0])-1):
					if(currentSpikeRate >= self.spikingThreshold and self.inputLayer[j].fired == 1):					#If both fire, increase weight
						currWeight = neuron.weights[j]
						deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
						if(currWeight+deltaW <= 1 and currWeight+deltaW>-1):
							neuron.weights[j] += deltaW
							# neuron.weights[j] = round(neuron.weights[j])
						elif(currWeight+deltaW == 1):
							neuron.weights[j] = 1.000
						# print("increased weight from ", currWeight, " to ", neuron.weights[j], " with delta ", (deltaW), " for input ", " ",numFired)
						numIncreased += 1
						neuron.numFired += 1
					elif (currentSpikeRate < self.spikingThreshold - 0.1 and self.inputLayer[j].fired == 1):			#if pre fires and post doesnt, decrease weight
						currWeight = neuron.weights[j]
						deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
						if(currWeight+deltaW >= -1):
							neuron.weights[j] -= deltaW
							# neuron.weights[j] = round(neuron.weights[j])
						elif(currWeight+deltaW == -1):
							neuron.weights[j] = -1.000
						neuron.notFired += 1
						# print("decreased weight from ", currWeight, " to ", neuron.weights[j], " with delta ", (-1*deltaW), " for input ", " ",numFired)
						numDecreased += 1
				if(neuron.numFired > neuron.notFired):
					neuron.fired = 1
					# print("Fired numFired:", neuron.numFired, " notFired: ", neuron.notFired)
				neuron.spikeRateForData.append(currentSpikeRate)									#Store current spike rate in array for training next
				# print(i,"\nNum increased: ",numIncreased, " Num Decreased:  ", numDecreased)
				# for j in range(len(input[0])):
			# print(neuron.weights,"\nNum increased: ",numIncreased, " Num Decreased:  ", numDecreased, " for input ")
			neuron.totalSpikeRate = totalSpikeRate/4
		print("Training layer 2...\n\n")
		for k in range(len(self.middleLayer[1])):
			neuron = self.middleLayer[1][k]
			currentSpikeRate = 0
			totalSpikeRate = 0
			numFired = 0

			for i in range(len(input)):
				totalSpikeRate += currentSpikeRate
				# preSpikeRate = 
				currentSpikeRate = 0
				for j in range(len(self.middleLayer[0])):
					multiplier = 1
					if(input[i][138] == 0):
						multiplier *= 0.8
					currentSpikeRate += neuron.runNeuron(multiplier*(self.middleLayer[0][j].spikeRateForData[i])*neuron.weights[j]*1.55)
				# print("Curr spike rate: ", currentSpikeRate)
				for j in range(len(self.middleLayer[0])):
					if(currentSpikeRate >= self.spikingThreshold and self.inputLayer[j].fired == 1):					#If both fire, increase weight
						currWeight = neuron.weights[j]
						deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
						if(currWeight+deltaW <= 1 and currWeight+deltaW>-1):
							neuron.weights[j] += deltaW
							# neuron.weights[j] = round(neuron.weights[j])
						elif(currWeight+deltaW == 1):
							neuron.weights[j] = 1.000
						# print("increased weight from ", currWeight, " to ", neuron.weights[j], " with delta ", (deltaW), " for input ", " ",numFired)
						numIncreased += 1
						neuron.numFired+=1
					elif (currentSpikeRate < self.spikingThreshold - 0.1 and self.inputLayer[j].fired == 1):			#if pre fires and post doesnt, decrease weight
						currWeight = neuron.weights[j]
						deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
						if(currWeight+deltaW >= -1):
							neuron.weights[j] -= deltaW
							# neuron.weights[j] = round(neuron.weights[j])
						elif(currWeight+deltaW == -1):
							neuron.weights[j] = -1.000
						neuron.notFired += 1
						# print("decreased weight from ", currWeight, " to ", neuron.weights[j], " with delta ", (-1*deltaW), " for input ", " ",numFired)
						numDecreased += 1
				if(neuron.numFired > neuron.notFired):
					neuron.fired = 1
					# print("Fired numFired:", neuron.numFired, " notFired: ", neuron.notFired)
				neuron.spikeRateForData.append(currentSpikeRate)									#Store current spike rate in array for training next
				# print(i,"\nNum increased: ",numIncreased, " Num Decreased:  ", numDecreased)
				# for j in range(len(input[0])):
			# print(neuron.weights,"\nNum increased: ",numIncreased, " Num Decreased:  ", numDecreased, " for input ")
			neuron.totalSpikeRate = totalSpikeRate/4
		print("Training layer 3...\n\n")
		for k in range(len(self.middleLayer[2])):
			neuron = self.middleLayer[2][k]
			currentSpikeRate = 0
			totalSpikeRate = 0
			numFired = 0

			for i in range(len(input)):
				totalSpikeRate += currentSpikeRate
				# preSpikeRate = 
				currentSpikeRate = 0
				for j in range(len(self.middleLayer[1])):
					multiplier = 1
					if(input[i][138] == 0):
						multiplier *= 0.8
					currentSpikeRate += neuron.runNeuron(multiplier*(self.middleLayer[1][j].spikeRateForData[i])*neuron.weights[j]*1.55)
				# print("Curr spike rate: ", currentSpikeRate)
				for j in range(len(self.middleLayer[1])):
					if(currentSpikeRate >= self.spikingThreshold and self.inputLayer[j].fired == 1):					#If both fire, increase weight
						currWeight = neuron.weights[j]
						deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
						if(currWeight+deltaW <= 1 and currWeight+deltaW>-1):
							neuron.weights[j] += deltaW
							# neuron.weights[j] = round(neuron.weights[j])
						elif(currWeight+deltaW == 1):
							neuron.weights[j] = 1.000
						# print("increased weight from ", currWeight, " to ", neuron.weights[j], " with delta ", (deltaW), " for input ", " ",numFired)
						numIncreased += 1
						neuron.numFired+=1
					elif (currentSpikeRate < self.spikingThreshold - 0.1 and self.inputLayer[j].fired == 1):			#if pre fires and post doesnt, decrease weight
						currWeight = neuron.weights[j]
						deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
						if(currWeight+deltaW >= -1):
							neuron.weights[j] -= deltaW
							# neuron.weights[j] = round(neuron.weights[j])
						elif(currWeight+deltaW == -1):
							neuron.weights[j] = -1.000
						neuron.notFired += 1
						# print("decreased weight from ", currWeight, " to ", neuron.weights[j], " with delta ", (-1*deltaW), " for input ", " ",numFired)
						numDecreased += 1
				if(neuron.numFired > neuron.notFired):
					neuron.fired = 1
					# print("Fired numFired:", neuron.numFired, " notFired: ", neuron.notFired)
				neuron.spikeRateForData.append(currentSpikeRate)									#Store current spike rate in array for training next
				# print(i,"\nNum increased: ",numIncreased, " Num Decreased:  ", numDecreased)
				# for j in range(len(input[0])):
			# print(neuron.weights,"\nNum increased: ",numIncreased, " Num Decreased:  ", numDecreased, " for input ")
			neuron.totalSpikeRate = totalSpikeRate/4
			# print("\n",totalSpikeRate/4,"\n")
		print("Training output neuron...\n\n")
		self.trainExcitatoryNeurons(input)
		self.trainInhibitoryNeurons(input)

	def saveWeights(self):
		layer1 = []
		for i in range(len(self.middleLayer[0])):
			currArray = []
			for j in range(len(self.middleLayer[0][i].weights)):
				currArray.append(self.middleLayer[0][i].weights[j])
			layer1.append(currArray)

		layer1 = np.array(layer1)

		np.savetxt('layer13.csv', layer1, delimiter=",")

		layer2 = []
		for i in range(len(self.middleLayer[1])):
			currArray = []
			for j in range(len(self.middleLayer[1][i].weights)):
				currArray.append(self.middleLayer[1][i].weights[j])
			layer2.append(currArray)
		
		layer2 = np.array(layer2)
		np.savetxt('layer23.csv', layer2, delimiter=",")

		layer3 = []
		for i in range(len(self.middleLayer[2])):
			currArray = []
			for j in range(len(self.middleLayer[2][i].weights)):
				currArray.append(self.middleLayer[2][i].weights[j])
			layer3.append(currArray)

		layer3 = np.array(layer3)
		np.savetxt('layer33.csv', layer3, delimiter=",")

		outputLayer = []
		for i in range(len(self.outputLayer.weights)):
			outputLayer.append(self.outputLayer.weights[i])

		outputLayer = np.array(outputLayer)		
		np.savetxt('outputLayer3.csv', outputLayer, delimiter=",")

	def getWeights(self):
		layer1 = genfromtxt('layer1.csv', delimiter=',')
		for i in range(len(self.middleLayer[0])):
			for j in range(138):
				self.middleLayer[0][i].weights[j] = layer1[i][j]

		layer2 = genfromtxt('layer2.csv', delimiter=',')
		for i in range(len(self.middleLayer[1])):
			for j in range(len(self.middleLayer[1][0].weights)):
				self.middleLayer[1][i].weights[j] = layer2[i][j]

		layer3 = genfromtxt('layer3.csv', delimiter=',')
		for i in range(len(self.middleLayer[2])):
			for j in range(len(self.middleLayer[2][0].weights)):
				self.middleLayer[2][i].weights[j] = layer3[i][j]

		outputLayer = genfromtxt('outputLayer.csv', delimiter=',')
		for i in range(len(self.outputLayer.weights)):
			self.outputLayer.weights[i] = outputLayer[i]

	def trainExcitatoryNeurons(self, input):
		for k in range(int(math.floor(len(self.outputLayer.weights)/2))):
			# print("Classification rate: ",self.middleLayer[k].spikeRateForData)
			currentSpikeRate = 0
			totalSpikeRate = 0
			numFired = 0

			for i in range(len(input)):
				preSpikeRate = self.middleLayer[2][k].spikeRateForData[i]
				preActivity = 1 if preSpikeRate >= self.spikingThreshold else 0
				currWeight = self.outputLayer.weights[k]
			
				# print("\nCurrSpikeRate: ",currSpikeRate, " preSpikeRate ", preSpikeRate)
				if(self.dataList[i][138] == 1):
					currSpikeRate = self.outputLayer.runNeuron(50)
				else:
					currSpikeRate = (self.outputLayer.runNeuron(preActivity*currWeight*20))
				# print("CurrSpikeRate: ",currSpikeRate, " preSpikeRate ", preSpikeRate)
				if preSpikeRate >= self.spikingThreshold and (self.dataList[i][138] == 1):
					currWeight = self.outputLayer.weights[k]
					deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
					if (self.dataList[i][138] == 1):
						deltaW = math.fabs(deltaW)*2
					if(currWeight+deltaW <=1):
						self.outputLayer.weights[k] += deltaW
						self.outputLayer.weights[k] = round(self.outputLayer.weights[k])
					else:
						self.outputLayer.weights[k] = 1.000
					# print("increased weight from ", currWeight, " to ", self.outputLayer.weights[k], " with delta ", round(deltaW), " for input ", input[i], " ")
				elif preSpikeRate >= self.spikingThreshold and self.dataList[i][138] == 0:
					currWeight = self.outputLayer.weights[k]
					deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
					if(currWeight-deltaW >=-1):
						self.outputLayer.weights[k] -= deltaW
						self.outputLayer.weights[k] = round(self.outputLayer.weights[k])
					else:
						neuron.weights[j] = -1.000
					# print("decreased weight from ", currWeight, " to ", self.outputLayer.weights[k], " with delta ", round(deltaW), " for input ", input[i], " ")
				# print("Weight for excitatory output ", input[i][138], " = ", self.outputLayer.weights)

	def trainInhibitoryNeurons(self, input):
		for k in range(int(math.floor(len(self.outputLayer.weights)/2)), self.hiddenLayerNum):
			currentSpikeRate = 0
			totalSpikeRate = 0
			numFired = 0
			for i in range(len(input)):
				preSpikeRate = self.middleLayer[2][k].spikeRateForData[i]
				preActivity = 1 if preSpikeRate >= self.spikingThreshold else 0
				currWeight = self.outputLayer.weights[k]
			
				currSpikeRate += (self.outputLayer.runNeuron(preActivity*currWeight*20))
				if preSpikeRate >= self.spikingThreshold and self.dataList[i][138] == 0:
					currWeight = self.outputLayer.weights[k]
					deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
					if(currWeight-deltaW >=-1):
						self.outputLayer.weights[k] -= deltaW
						self.outputLayer.weights[k] = round(self.outputLayer.weights[k])
					else:
						self.outputLayer.weights[k] = -1.000
					# print("decreased weight from ", currWeight, " to ", self.outputLayer.weights[k], " with delta ", round(deltaW), " for input ", input[i], " ")
				elif preSpikeRate >= self.spikingThreshold and self.dataList[i][138] == 1:
					currWeight = self.outputLayer.weights[k]
					deltaW = (self.learningRate * 1 * (1 - 1*currWeight))/self.timeStep
					if(currWeight+deltaW <=1):
						self.outputLayer.weights[k] += deltaW
						self.outputLayer.weights[k] = round(self.outputLayer.weights[k])
					else:
						self.outputLayer.weights[k] = 1.000
					# print("increased weight from ", currWeight, " to ", self.outputLayer.weights[k], " with delta ", round(deltaW), " for input ", input[i], " ")
				# print("Weight for inhibitory output ", input[i][138], " = ", self.outputLayer.weights)

	def classify(self, inputs):
		correctlyClassified = 0
		incorrectlyClassified = 0
		total = len(inputs)
		firingRates = []

		for x in range(len(inputs)):
			input = inputs[x]
			currGenre = input[x][138]
			for k in range(len(self.inputLayer)):
				neuron = self.inputLayer[k]
				currSpikeRate = 0
				for i in range(len(input)):
					currSpikeRate += neuron.runNeuron(input[i][k]*5+7.9)
				neuron.classificationRate = currSpikeRate

			for i in range(len(self.inputLayer)):
				neuron = self.inputLayer[i]
				currSpikeRate = 0
				currActivity = 1 if neuron.classificationRate > self.spikingThreshold else 0
				neuron.classificationActivity = currActivity

			#Layer 1
			for k in range(len(self.middleLayer[0])):
				neuron = self.middleLayer[0][k]
				currSpikeRate = 0
				multiplier = 0.7
				if(self.inputLayer[k].classificationActivity == 0):
					multiplier = 0.8
				for i in range(len(self.middleLayer[0][k].weights)):
					currSpikeRate += neuron.runNeuron(multiplier*neuron.weights[k]*self.inputLayer[k].classificationRate*2.0)
				neuron.classificationRate = currSpikeRate
				# print("Layer 1: ", currSpikeRate)

			for i in range(len(self.middleLayer[0])):
				neuron = self.middleLayer[0][i]
				currSpikeRate = 0
				currActivity = 1 if neuron.classificationRate > self.spikingThreshold else 0
				neuron.classificationActivity = currActivity
			#layer 2
			for k in range(len(self.middleLayer[1])):
				neuron = self.middleLayer[1][k]
				currSpikeRate = 0
				multiplier = 0.9
				for i in range(len(self.middleLayer[1][k].weights)):
					if(self.middleLayer[0][i].classificationActivity == 0 or input[0][138] == 0):
						multiplier = 0.5
					currSpikeRate += neuron.runNeuron(multiplier*neuron.weights[i]*self.middleLayer[0][i].classificationRate*1.9)
				neuron.classificationRate = currSpikeRate
				# print("Layer 2: ", currSpikeRate)

			for i in range(len(self.middleLayer[1])):
				neuron = self.middleLayer[1][i]
				currSpikeRate = 0
				currActivity = 1 if neuron.classificationRate > self.spikingThreshold else 0
				neuron.classificationActivity = currActivity


			#layer 3
			for k in range(len(self.middleLayer[2])):
				neuron = self.middleLayer[2][k]
				currSpikeRate = 0
				multiplier = 0.7
				for i in range(len(self.middleLayer[1][k].weights)):
					if(self.middleLayer[1][i].classificationActivity == 0 or input[0][138] == 0):
						multiplier = 0.5
					currSpikeRate += neuron.runNeuron(multiplier*neuron.weights[i]*self.middleLayer[1][i].classificationRate*2.1)
				neuron.classificationRate = currSpikeRate
				# print("Layer 3: ", currSpikeRate)

			for i in range(len(self.middleLayer[2])):
				neuron = self.middleLayer[2][i]
				currSpikeRate = 0
				currActivity = 1 if neuron.classificationRate > self.spikingThreshold else 0
				neuron.classificationActivity = currActivity

			#output layer
			outputSpikingRate = 0
			currSpikeRate = 0
			multiplier = 0.7
			for i in range(len(self.middleLayer[2])):
				if(self.middleLayer[2][i].classificationActivity == 0  or input[0][138] == 0):
					multiplier = 0.75
				currSpikeRate += self.outputLayer.runNeuron(multiplier*self.outputLayer.weights[i]*self.middleLayer[2][i].classificationRate*1.7)
			outputSpikingRate = currSpikeRate

			print("Ouput firing rate: ",outputSpikingRate," for genre ", currGenre)

	def round(input):
		return math.ceil(input*100000)/100000



test = GenreClassifier()
test.getMetalFiles()
test.getClassicalFiles()
np.random.shuffle(test.dataList)
# print("Reading file: \n\n")
# test.dataList = np.genfromtxt("global.csv", delimiter=',')
print(test.dataList, "\nShape: ", test.dataList.shape)
# test.train()
# test.saveWeights()
test.getWeights()
test.isFirst = 0
classificationInput = np.concatenate((test.getClassificationMetalInput(), test.getClassificationClassicalInput()),axis=0)
np.random.shuffle(classificationInput)
test.classify(classificationInput)
# test.train()
# test.train()
# test.classify()


