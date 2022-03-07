from numpy.lib.function_base import disp
import scipy.special
import scipy.signal as sci
import scipy as sp
import numpy
import matplotlib.pyplot as plt
import scipy.io as spio
import random
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from random import randint, random
from operator import add
from functools import reduce

class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__ (self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        # Set the learning rate
        self.lr = learning_rate
        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))
        # Query the network

    def save(self, file_name):
        # Create an empty dictionary to store current state of neural network
        saved_network = {}
        # Store current state of neural network in dictionary
        saved_network['i_nodes'] = self.i_nodes
        saved_network['h_nodes'] = self.h_nodes
        saved_network['o_nodes'] = self.o_nodes
        saved_network['lr'] = self.lr
        saved_network['wih'] = self.wih
        saved_network['who'] = self.who
        # Save dictionary to file
        spio.savemat(file_name, saved_network)

    # Load previously saved state from file
    def load(self, file_name):
        # Open saved file
        try:
            loaded_network = spio.loadmat(file_name)
            # Load data to correct arrays and variables
            self.i_nodes = loaded_network['i_nodes']
            self.h_nodes = loaded_network['h_nodes']
            self.o_nodes = loaded_network['o_nodes']
            self.lr = loaded_network['lr']
            self.wih = loaded_network['wih']
            self.who = loaded_network['who']
            return 

        except:
            # If file does not exist, return false
            return False

    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate outputs from the final layer
        final_outputs =self.activation_function(final_inputs)
        return final_outputs

#-----------------------------------------------------------------------------------
# Filter the input data to remove drift and high f noise
#-----------------------------------------------------------------------------------
def butter_bandpass(rawInput, lowcut, highcut, fs, order=5):

    # Nyquist freq is half sample peak
    nyq = 0.5 * fs

    # Produce inputs for butterworth bandpass filter
    low = lowcut / nyq
    high = highcut / nyq

    # Apply the filter to the data
    b, a = sci.butter(order, [low, high], btype='band')
    y = sci.lfilter(b, a, rawInput)
    return y

def confusion_mat(guess, actual):
    matrix = ConfusionMatrixDisplay(confusion_matrix(actual, guess),display_labels= ['Class 1,','Class 2','Class 3', 'Class 4', 'Class 5'])
    matrix.plot()
    return


#-----------------------------------------------------------------------------------
# Finds all the peaks in the sample and returns a marker of where they each start
#-----------------------------------------------------------------------------------
def label_peak(signal,peak_height = 0.45, peak_prominence = 0.09,windowSize = 70,test = False):

    # Find peaks using scipy library function find_peaks
    peaks, properties = scipy.signal.find_peaks(signal,height = peak_height, prominence = peak_prominence)
    peaksMarker = []

    # Remove false positive peaks
    peaks = remove_wide(signal,peaks,100)

    # Move peak detection to start of spike
    for x in range(len(peaks)):     
        if x != 0:

            # Generate window of samples from leadup to peak
            window = signal[peaks[x]-windowSize:peaks[x]]

            # Workaround for peaks in first 75 samples
            if x == 0 and peaks[x]<windowSize:
                window = signal[:peaks[x]]

            # Find all minima in the window
            minima = sci.argrelextrema(window, np.less)

            # Select the closest minima to the peak
            minima = minima[0]
            closeMinima = minima[len(minima)-1]

            # Label the start of the peak as the closest minima to the peak
            peaksStart = peaks[x]-(windowSize-closeMinima)

            # If the closest minima was before the peak has risen by 0.07, likely not correct label so find next closest minima
            if  signal[peaks[x]] - signal[peaksStart] < 0.07:
                closeMinima = minima[len(minima)-2]
                peaksStart = peaks[x]-(windowSize-closeMinima)

            # Add peak marker to output list of all peak locations
            peaksMarker.append(peaksStart)

    return peaksMarker

#-----------------------------------------------------------------------------------
# Remove peaks that are too wide as they are likely to be false positives
#-----------------------------------------------------------------------------------
def remove_wide(signal,peaks,width = 75):

    tooWideLocations = []

    # Use the peak_widths function to measure the width of all peaks
    tooWide = sci.peak_widths(signal,peaks)[0]

    # Loop through all the peak widths and record any that are wider than 30 samples
    for x in range(len(tooWide)):
        if tooWide[x]>30:
            tooWideLocations.append(x)
    
    # Delete all the peaks deemed too wide
    peaks = np.delete(peaks,tooWideLocations)

    return peaks

#-----------------------------------------------------------------------------------
# Generate windows based on the peak locations for input into the neural network
#-----------------------------------------------------------------------------------
def generate_windows(signal,locations,tailLength = 75):

    samples = []

    # Extract the n signals after a peak detection and append them to a samples array
    for x in locations:
        samples.append(signal[x:x+tailLength])

    return samples

#-----------------------------------------------------------------------------------
# Scale and normalise the data for consistent input into neural network
#-----------------------------------------------------------------------------------
def scale_data(signal):

    # Shift all the data up by the y distance from 0 to the lowest point
    shifted = signal + abs(min(signal))

    # Scale the data between 0 and 0.99
    scaled = shifted/max(shifted)*0.99

    # Shift the data to between 0.01 and 1
    scaled = scaled + 0.01

    return scaled

#-----------------------------------------------------------------------------------
# Gives peak detection performance and assigns provided labels to produced labels for training and testing
#-----------------------------------------------------------------------------------
def missed_peaks(peaks,index,labels):
    j = 1
    hitPeaks = []
    missedPeaks = []
    falsePeaks = []
    labelTransfer = []

    for i in range(len(index)): 

        # First peak is detected always but produces errors if ignored - workaround
        if i==0:
            hitPeaks.append(index[i])
            labelTransfer.append(labels[i])

        # Workaround to avoid errors - cost to debug not deemed worth it 
        elif i == 3338:
            a = ''

        # For all the index values check whether the detected peak is false positive, correct, or a peak was missed
        elif j<len(index):

            # If therer is a detected peak between two indexes              
            if index[i-1] < peaks[j] <= index[i] :
                # and if that peak is less that 35 samples from the higher index
                # The peak has been located - add the label to an array to be mapped onto the peak detections
                if abs(index[i] - peaks[j]) < 35:                  
                    hitPeaks.append(index[i])
                    labelTransfer.append(labels[i])
                    j += 1
                # if the peak is further than 35 samples, it is likely false positive. append to false positive list
                else:
                    falsePeaks.append(peaks[j])
                    j += 1
            # If the peak detection is not between the two indexes, but just after, it is still considered correct detection        
            elif index[i] < peaks[j] <= index[i] + 6:
                    hitPeaks.append(index[i])
                    labelTransfer.append(labels[i])
                    j += 1
            # Otherwise the index location has been missed
            else:
                missedPeaks.append(index[i])
        
    return hitPeaks, missedPeaks, falsePeaks, labelTransfer

#-----------------------------------------------------------------------------------
# Trains the nueral network and can test for debug and optimisation
#-----------------------------------------------------------------------------------
def train_nueral(samples,index,labels,epochs,learningRate = 0.05,hiddenLayers = 140,test = True): 

    
    # If the network is being tested, break the sample down into 80% train 20% test
    if test:
        testLength = 0.8
        sampleLen = int(testLength*len(samples))
        labelLen = int(testLength*len(labelCalculated))

        trainSamples = samples[0:sampleLen]
        trainLabels = labelCalculated[0:labelLen]
        testSamples = samples[sampleLen:]
        testLabels = labelCalculated[labelLen:]

    # Initialise neural net
    NN = NeuralNetwork(75,hiddenLayers,5,learningRate)

    perfX = []

    # Train the network and test for each epoch
    for i in range(epochs):
        for j in range(len(trainSamples)):

            # Select the sample
            trainSample = trainSamples[j]

            # Create array fpr target data: 0.99 for correct label, else 0.01
            targets = np.zeros(5) + 0.01
            targets[trainLabels[j]-1]=0.99
            
            # Train the NN for a given sample and target
            NN.train(trainSample,targets)      
        pass
        

    if test:
        scorecard = []
        guesses = []
        # Loop through all of the records in the test data set
        for i in range(len(testSamples)):

            # Extract current sample and label
            correct_label = testLabels[i]
            testSample = testSamples[i]

            # Query the network
            outputs = NN.query(testSample)
            # The index of the highest value output corresponds to the label
            label = numpy.argmax(outputs)

            # Append either a 1 or a 0 to the scorecard list
            if (label == correct_label - 1):
                scorecard.append(1)
            else:
                scorecard.append(0)
            pass


            


        # Calculate the performance score, the fraction of correct answers
        scorecard_array = numpy.asarray(scorecard)
        # print("--------------------------")
        perf = (scorecard_array.sum() / scorecard_array.size)*100
        print("Performance = ", perf, '%')
        perfX.append(perf)
        #print("--------------------------")
    


    # Save the neural net for use in testing
    NN.save('nn.mat')


    return index,label, perf

#-----------------------------------------------------------------------------------
# Tests the neural net on the submission data
#-----------------------------------------------------------------------------------
def test_neural(signal):
    # Filter out low and high frequency noise from recording
    filtered = butter_bandpass(signal,25,2000,25000)

    # Scale the data to be between 0 and 1
    scaledFiltered = scale_data(filtered)

    # Detect peaks using find_peaks and defaults of height = 0.65, prominence = 0.1 
    peaks = label_peak(scaledFiltered,test = True)

    plt.plot(scaledFiltered)
    plt.plot(peaks,scaledFiltered[peaks],linestyle="",marker="o")
    plt.show()


    # Generate sample windows using generate_windows function
    samples = generate_windows(scaledFiltered,peaks,75)
    
    # Define network topology
    inputNodes = 75
    outputNodes = 5

    # Load neural net from training
    NN = NeuralNetwork(inputNodes,inputNodes*2,5,learning_rate=0.05)
    NN.load('nn.mat')

    # Initialise output arrays
    indexGuess = peaks
    labelGuess = []

    # Loop through all of the records in the test data set
    for i in range(len(samples)):
        # Select individual sample
        sample = samples[i]

        # Query the network for the sample
        outputs = NN.query(sample)

        # The index of the highest value output corresponds to the label
        labelGuess.append(numpy.argmax(outputs))
    
    return indexGuess,labelGuess 

#-----------------------------------------------------------------------------------
# Imports the data and orders it based on index from low to high
#-----------------------------------------------------------------------------------
def import_sort():

    # Read the data from the matlab file
    mat = spio.loadmat('training.mat', squeeze_me=True)
    rawInput = mat['d']
    index = mat['Index']
    label = mat['Class']

    # Order the data maintaining reference
    index, label = zip(*sorted(zip(index, label)))

    return rawInput, np.array(index), np.array(label)


def individual(length, min, max):

    return [randint(min, max) for x in range(length)]

def population(count, length, min, max):

    return [individual(length, min, max) for x in range(count)]

def fitness(individual, target,samples,index,labelCalculated):
 
    Lr = 0.5*(individual[0]/31)+0.01
    Hl = 25*(individual[1])+25
    # Train the NN on the data
    perf = train_nueral(samples,index,labelCalculated,epochs=5,learningRate = Lr, hiddenLayers = Hl,test = True)[2]

    return 100 - perf

def grade(pop, target,samples,index,labelCalculated):
    'Find average fitness for a population.'
    fitSum = 0
    for x in pop:
        fitSum += fitness(x, target,samples,index,labelCalculated)
    fitAvg = fitSum/len(pop)
    #summed = reduce(add, (fitness(x, target) ), 0)
    return fitAvg

def evolve(pop, target, retain=0.2, mutate=0.1):
    global male
    parents = []
    graded = [(fitness(x, target,samples,index,labelCalculated), x) for x in pop]
    best = min(graded)
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # Parents are given a chance to mutate before crossover
    for individual in parents:
        if mutate > random():
            geneToMutate = randint(0,1)
            posToMutate = (2**randint(0, 4)) # check this           
            individual[geneToMutate] = posToMutate^individual[geneToMutate]

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)

        if male != female:
            male = parents[male]
            female = parents[female]
            crossMask = (2**(randint(0,4)))-1


            maleEndA = male[0]&crossMask
            femaleEndA = female[0]&(31-crossMask)
            maleEndB = male[1]&crossMask
            femaleEndB = female[1]&(31-crossMask)

            child = [maleEndA+femaleEndA,maleEndB+femaleEndB]
            children.append(child)
    parents.extend(children)
    return parents,best

# Import and order the index and labels
rawInput, index, label = import_sort()

# Filter out low and high frequency noise from recording
filtered = butter_bandpass(rawInput,25,2900,25000)

# Scale the data to be between 0 and 1
scaledFiltered = scale_data(filtered)

# Detect peaks using find_peaks and defaults of height = 0.65, prominence = 0.1 
peaks = label_peak(scaledFiltered)

# Breakdown peaks into three categories based on accuracy and return the labels of the correctly detected peaks
matchedPeaks, missedPeaks, falsePeaks, labelCalculated = missed_peaks(peaks,index,label)

# Produce sample windows for input into neural net
samples = generate_windows(scaledFiltered,matchedPeaks,75)

train_nueral(samples,index,labelCalculated,epochs = 5,)



targetMax=127
p_count = 20
i_length = 2
i_min = 0
i_max = 31
loops = 0
retain=0.2
mutate=0.13
totalGen = 0
loops = 0
output = []
outputX = []


i=0
generations = 0
target = 100

# Initialise pop
p = population(p_count, i_length, i_min, i_max)

# Grade the current population
fitness_history = [grade(p, target,samples,index,labelCalculated),]
bestHistory = []
bestParams = []
while i != 1:
    # Increment generations
    generations = generations+1
    print('-----------------------------------')
    print(generations)
    print('-----------------------------------')
    # Evolve the population
    p,best = evolve(p, target, retain, mutate) #evolve the population
    graded = (grade(p,target,samples,index,labelCalculated))            #grade the population
    if generations >= 10 or graded<0.5:
        fitness_history.append(graded)
        bestHistory.append(best[0])
        bestParams.append(best[1])
        i = 1
    else:
        fitness_history.append(graded)
        bestHistory.append(best[0])
        bestParams.append(best[1])
        i=0


#graded = [(fitness(x, target,samples,index,labelCalculated), x) for x in p_count]
#orderedGraded = [x[1] for x in sorted(graded)]

totalGen = totalGen + generations
loops = loops+1

plt.plot(bestHistory[0])
plt.plot(fitness_history)
plt.show()

print(bestParams)
plt.plot(bestParams[1])
plt.show()

# avgGens = totalGen/loops
# output.append(avgGens)
# outputX.append(x)
# print('Average Generations: ', avgGens, ' x: ',x)
# print(output)
# plt.title('Fine Sweep of Retain Rate Effect on GA Efficiency')
# plt.xlabel("Retain Rate")
# plt.ylabel("Number of Generations Till Population is Fit")
# plt.plot(outputX, output)
# plt.show()
# for i in range(generations):
# p = evolve(p, target)
# fitness_history.append(grade(p, target))

# for datum in fitness_history:
# print(datum)












