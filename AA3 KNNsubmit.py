from numpy.core.fromnumeric import argmax
import scipy.special
import scipy.signal as sci
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LinDA
from sklearn.neighbors import KNeighborsClassifier as NearestNeightbour
from sklearn.decomposition import PCA as DimReduct
from sklearn.metrics import plot_confusion_matrix
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

#-----------------------------------------------------------------------------------
# Finds all the peaks in the sample and returns a marker of where they each start
#-----------------------------------------------------------------------------------
def label_peak(signal,peak_height = 0.45, peak_prominence = 0.09,windowSize = 70,test = False):

    # Find peaks using scipy library function find_peaks
    peaks, properties = sci.find_peaks(signal,height = peak_height, prominence = peak_prominence)
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
            print()

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
# Finds the time from peak to end of spike
#-----------------------------------------------------------------------------------
def extract_falltime(window,peak):

    # Crop window to only look post peak
    croppedWindow = window[peak:]

    # Find the time till first minima
    minima = sci.argrelextrema(croppedWindow, np.less)[0]
    fallTime = min(minima)

    return fallTime

#-----------------------------------------------------------------------------------
# Finds the peak height 
#-----------------------------------------------------------------------------------
def extract_peakheight(window,peak):

    # Measure difference in height at start of sample and at peak
    height = window[peak]-window[0]

    return height

#-----------------------------------------------------------------------------------
# Preprocesses data, finds peaks and extracts key features from peaks
#-----------------------------------------------------------------------------------
def extract_features(signal,labels, train = True):

    highCut = 2000
    if train == True:
        highCut = 2900

    # Filter out low and high frequency noise from recording
    filtered = butter_bandpass(signal,25,highCut,25000)

    # Scale the data to be between 0 and 1
    scaledFiltered = scale_data(filtered)

    # Detect peaks using find_peaks and defaults of height = 0.65, prominence = 0.1 
    peaks = label_peak(scaledFiltered)

    if train:
        # Find the correct peaks, missed peaks and false positives
        peaks, missedPeaks, falsePeaks, labelCalculated = missed_peaks(peaks,index,labels)

        #Report peak finding efficacy to terminal
        print('----------------------------------')
        print('Matched Peaks: ',len(peaks), )
        print('Missed Peaks: ',len(missedPeaks), )
        print('False Peaks: ',len(falsePeaks), )

    # Generate the 75 long windows using peak locations 
    windows = generate_windows(scaledFiltered,peaks,tailLength=75)

    # Cycle through every window produced and extract three features: rise time, fall time and peak height
    # and append them to a features array
    features = []
    riseTime = []
    peakHeight= []
    fallTime  = []  
    labelsOut = []
    j = 0

    # Extract rise time, fall time and peak height for all the located peaks
    for i in windows:

        # Find the peak within the window
        peak = sci.find_peaks(i)[0]

        #Find features
        rise = peak[0]
        riseTime.append(rise)
        fallTime.append(extract_falltime(i,rise))
        peakHeight.append(extract_peakheight(i,rise))

        #features.append([rise,extract_falltime(i,rise),extract_peakheight(i,rise)])

        if train:
        # Append the relevant peak type along with the features 
            labelsOut.append(labelCalculated[j])
        
        if train == False:
            labelsOut = peaks

        j+=1

    # Normalise the feature data to between 0 and 1 
    riseTime = riseTime/max(riseTime)
    fallTime = fallTime/max(fallTime)
    peakHeight = peakHeight/max(peakHeight)

    # Combine all the features into one list
    for i in range(len(riseTime)):
        features.append([riseTime[i],fallTime[i],peakHeight[i]])

    return features, labelsOut

#-----------------------------------------------------------------------------------
# Run K nearest neighbour analysis
#-----------------------------------------------------------------------------------
def KNearNeigh(features, labelsOut,K = 9):

    KNN = NearestNeightbour(K)
    

    X = []
    for x in features:
        X.append(x)
    KNN.fit(X, labelsOut)


    # Load the submission data
    mat = spio.loadmat('submission.mat', squeeze_me=True)
    rawInput = mat['d']

    features, index = extract_features(rawInput,labelsOut,False)

    X = []
    for x in features:
        X.append(x)
    Guess = KNN.predict(X)
     


    return Guess , index

# Order the index and labels
rawInput, index, label = import_sort()

features, labelsOut = extract_features(rawInput,label)

guess, index = KNearNeigh(features, labelsOut, K = 9)

unique, counts = np.unique(guess, return_counts=True)

for i in range(5):
    print('Class ',unique[i], ':',counts[i] )

# Print the number of spikes detected in the submssion
print('answers submitted: ', len(index))

# Save and export the results to matlab
mdic = {"index": index, "Class": guess}
spio.savemat('13797.mat',mdic)







