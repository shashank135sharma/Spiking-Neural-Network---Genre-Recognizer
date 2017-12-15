import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pylab import *
from random import *
import math

class Neuron:
	def __init__(self, durationForSimulation, numInputs):
		self.T = durationForSimulation #Time to simulate, in ms
		self.dT = 0.1 #the dT time step in dV/dT
		self.VArray = zeros(int((self.T/self.dT) + 1)) #Array of membrane potentials, for plotting later
		self.Vt = 1 #Threshold, in V
		self.Vr = 0 #Reset potential, in mV
		self.initialV = 0 #Initial Membrane Potential = Formula is change in mV
		self.R = 1 #Membrane resistance, in kOhms
		self.C = 10 #Capacitance in uF
		self.tauM = self.R*self.C #Membrane time constant, in miliseconds
		self.firingRate = 10

		self.currentChangeInPotential = float(10.0) #Change in current membrane potential - Used in array

		self.numFired = 0
		self.notFired = 0
		self.fired = 0
		self.spikeRateForData = []
		self.totalSpikingRate = 0

		self.classificationRate = 0
		self.classificationActivity = 0

		self.weights = [];
		for i in range(numInputs):
			randomWeight = math.ceil(uniform(0,2000)-1000)/1000
			self.weights.append(randomWeight)

	def runNeuron(self, inputCurrent):
		self.counter = 0.0
		I = inputCurrent #input current, in Amps - Given by parameter, plus minimum threshold to actually fire

		spikeSum = 0.
		self.VArray = zeros(int((self.T/self.dT) + 1)) #Array of membrane potentials, for plotting later
		self.currentChangeInPotential = 0

		for i in range(1, len(self.VArray)) :
			self.currentChangeInPotential = (-1*self.VArray[i-1] + self.R*I)
			self.VArray[i] = self.VArray[i-1] + self.currentChangeInPotential/self.tauM*self.dT
			if(self.VArray[i] >= self.Vt):
				self.VArray[i] = self.Vr
				spikeSum += 1

		return (math.ceil((spikeSum/self.T)*1000))/1000

	def showGraph(self):
		currTime = arange(0, self.T+self.dT, self.dT) 
		plt.plot(currTime, self.VArray)
		plt.xlabel('Time in miliseconds')
		plt.ylabel('Membrane Potential in Volts')
		plt.title('Simulation - LIF Neuron')
		plt.ylim([0,4])
		plt.show()
		print(spikeSum/self.T)

time = 20
sampleNeuronRun = Neuron(time, 5)
print("", 1.4, ": ", sampleNeuronRun.runNeuron(5))
# sampleNeuronRun = Neuron(time)
# print("", 1.4, ": ", sampleNeuronRun.RunNeuron(1.26))
# sampleNeuronRun = Neuron(time)
# print("", 1.4, ": ", sampleNeuronRun.RunNeuron(1.27))
# sampleNeuronRun = Neuron(time)
# print("", 1.5, ": ", sampleNeuronRun.RunNeuron(1.28))
# sampleNeuronRun = Neuron(time)
# print("", 1.5, ": ", sampleNeuronRun.RunNeuron(1.29))
# sampleNeuronRun = Neuron(time)
# print("", 1.5, ": ", sampleNeuronRun.RunNeuron(1.31))

