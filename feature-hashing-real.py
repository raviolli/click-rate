labVersion = 'cs190.1x-lab4-1.0.4'

from test_helper import Test

import glob
from io import BytesIO
import os.path
import tarfile
import urllib
import urlparse

from collections import defaultdict
import hashlib

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

import numpy as np
from math import log
from math import exp #  exp(-t) = e^-t
import matplotlib.pyplot as plt


baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)

#--------------------
# Load Data 
#--------------------

if os.path.isfile(fileName):
    rawData = (sc
               .textFile(fileName, 2)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data
    print rawData.take(1)


if 'rawData' in locals():
    print 'rawData already loaded.  Nothing to do.'
else:
    try:
        dacSample = sc.textFile("/FileStore/tables/nm4vfnu91473248456597/dac_sample.txt").collect()
        dacSample = [unicode(x.replace('\n', '').replace('\t', ',')) for x in dacSample]
        rawData  = (sc
                    .parallelize(dacSample, 1)  # Create an RDD
                    .zipWithIndex()  # Enumerate lines
                    .map(lambda (v, i): (i, v))  # Use line index as key
                    .partitionBy(2, lambda i: not (i < 50026))  # Match sc.textFile partitioning
                    .map(lambda (i, v): v))  # Remove index
        print 'rawData loaded from url'
        print rawData.take(1)
    except IOError:
        print 'Unable to unpack: {0}'.format(url)

#-------------------
# Function Code
#-------------------

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

def parsePoint(point):
    """Converts a comma separated string into a list of (featureID, value) tuples.

    Note:
        featureIDs should start at 0 and increase to the number of features - 1.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.

    Returns:
        list: A list of (featureID, value) tuples.
    """
    data = point.split(',')
    data = data[1:]
    ret=[]
    i=0
    for feature in data:
        ret.append( (i,feature) )
        i += 1

    return ret

def createOneHotDict(inputData):
    """Creates a one-hot-encoder dictionary based on the input data.

    Args:
        inputData (RDD of lists of (int, str)): An RDD of observations where each observation is
            made up of a list of (featureID, value) tuples.

    Returns:
        dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
            unique integers.
    """
    DistinctFeats = (inputData.flatMap(lambda data: data).distinct())
    dictFeats = DistinctFeats.zipWithIndex().collectAsMap()
   
    return dictFeats #returns key-value pair as dictionary

def parseOHEPoint(point, OHEDict, numOHEFeats):
    """Obtain the label and feature vector for this raw observation.

    Note:
        You must use the function `oneHotEncoding` in this implementation or later portions
        of this lab may not function as expected.

    Args:
        point (str): A comma separated string where the first value is the label and the rest
            are features.
        OHEDict (dict of (int, str) to int): Mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The number of unique features in the training dataset.

    Returns:
        LabeledPoint: Contains the label for the observation and the one-hot-encoding of the
            raw features based on the provided OHE dictionary.
    """
    
    data = point.split(',')
    rawDataFeats=[]
    i=0
    for feature in data[1:]:
        rawDataFeats.append( (i,feature) )
        i += 1


    #rawData = (label (category, featureValue), (,) ....)
    sparseFeats = oneHotEncoding(rawDataFeats, OHEDict, numOHEFeats)
    # sparseFeats = sparse Hot Encoding features
    return LabeledPoint(data[0], sparseFeats)   #(label, sparse Vector Feature)

def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        If a (featureID, value) tuple doesn't have a corresponding key in OHEDict it should be
        ignored.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        OHEDict (dict): A mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indices equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.  with values equal to 1.0.
    """

    uniqueFeatID=[]

    for item in rawFeats:
        if (OHEDict.has_key(item)): #only perform if feature is in the dictionary
            uniqueFeatID.append( OHEDict.get(item) )

    return SparseVector(numOHEFeats, sorted(uniqueFeatID), np.ones( len(uniqueFeatID) ))

def computeLogLoss(p, y):
    """Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.

    Returns:
        float: The log loss value.
    """
    epsilon = 10e-12
    loss=0;

    if (p==1):
        p=p-epsilon
    elif (p==0):
        p=p+epsilon

    if ( y == 1 ):
        loss = -log(p)
    elif ( y == 0 ):
        loss = -log(1-p)

    return loss

def getP(x, w, intercept):
    """Calculate the probability for an observation given a set of weights and intercept.

    Note:
        We'll bound our raw prediction between 20 and -20 for numerical purposes.

    Args:
        x (SparseVector): A vector with values of 1.0 for features that exist in this
            observation and 0.0 otherwise.
        w (DenseVector): A vector of weights (betas) for the model.
        intercept (float): The model's intercept.

    Returns:
        float: A probability between 0 and 1.
    """
    rawPrediction = x.dot(w) + intercept

    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return 1 / (1 + exp(-rawPrediction))

def evaluateResults(model, data):
    """Calculates the log loss for the data given the model.

    Args:
        model (LogisticRegressionModel): A trained logistic regression model.
        data (RDD of LabeledPoint): Labels and features for each observation.

    Returns:
        float: Log loss for the data.
    """
    labels = data.map(lambda featdata: featdata.label).zipWithIndex()
    labels = labels.map(lambda data: (data[1], data[0])) #(iterator, label)
    
    predictions = data.map(lambda featdata: getP(featdata.features, model.weights, model.intercept)).zipWithIndex()
    predictions = predictions.map(lambda data: (data[1], data[0]))

    dataPredictions = predictions.join(labels)

    logloss = dataPredictions.map(lambda data: computeLogLoss(data[1][0], data[1][1])).mean()
    return logloss

def hashFunction(numBuckets, rawFeats, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        numBuckets (int): Number of buckets to use as features.
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = {}
    for ind, category in rawFeats:
        featureString = str(category) + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)
    #return sparseFeatures

def parseHashPoint(point, numBuckets):
    """Create a LabeledPoint for this observation using hashing.

    Args:
        point (str): A comma separated string where the first value is the label and the rest are
            features.
        numBuckets: The number of buckets to hash to.

    Returns:
        LabeledPoint: A LabeledPoint with a label (0.0 or 1.0) and a SparseVector of hashed
            features.
    """
    

    data = point.split(',')
    label = data[0]
    rawDataFeats=[]
    i=0
    for feature in data[1:]:
        rawDataFeats.append( (i,feature) )
        i += 1


    #rawData = ((category, featureValue), (,) ....)
    hashDict = hashFunction(numBuckets, rawDataFeats, False)    #hashes the feature into bucket and provides dictionary

    keys = []
    values = []
    item = sorted(hashDict.items())

    for i in item:
        keys.append(i[0])
        values.append(i[1])

    sparseFeats = SparseVector(numBuckets, keys , values)
    # sparseFeats = sparse Hashed Features
    return LabeledPoint(label, sparseFeats)   #(label, sparsed Hashed Feature Vector)


def computeSparsity(data, d, n):
    """Calculates the average sparsity for the features in an RDD of LabeledPoints.

    Args:
        data (RDD of LabeledPoint): The LabeledPoints to use in the sparsity calculation.
        d (int): The total number of features.
        n (int): The number of observations in the RDD.

    Returns:
        float: The average of the ratio of features in a point to total features.
    """
    sparseAvg = (data.map(lambda lp: len(lp.features.values)/d)).sum() / n
    return sparseAvg

#-------------------
# Main Code
#-------------------

weights = [.8, .1, .1]
seed = 42
# Use randomSplit with weights and seed
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit(weights, seed)
# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()

nTrain = rawTrainData.count()
nVal = rawValidationData.count()
nTest = rawTestData.count()
print ('# Training Samples = {0:d}\t# Val Samples = {1:d}\t# Test Samples = {2:d}').format(nTrain, nVal, nTest)


parsedTrainFeat = rawTrainData.map(parsePoint)

numCategories = (parsedTrainFeat
                 .flatMap(lambda x: x)
                 .distinct()
                 .map(lambda x: (x[0], 1))
                 .reduceByKey(lambda x, y: x + y)
                 .sortByKey()
                 .collect())

ctrOHEDict = createOneHotDict(parsedTrainFeat)
numCtrOHEFeats = len(ctrOHEDict.keys())

OHETrainData = rawTrainData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))
OHETrainData.cache()

# Check that oneHotEncoding function was used in parseOHEPoint
backupOneHot = oneHotEncoding
oneHotEncoding = None
withOneHot = False
try: parseOHEPoint(rawTrainData.take(1)[0], ctrOHEDict, numCtrOHEFeats)
except TypeError: withOneHot = True
oneHotEncoding = backupOneHot

OHEValidationData = rawValidationData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))
OHEValidationData.cache()

#--------
# Regression Model (machine learning)
#--------

# fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True

model0 = LogisticRegressionWithSGD.train(OHETrainData, iterations=numIters, step=stepSize, regParam=regParam,regType=regType, intercept=includeIntercept)

sortedWeights = sorted(model0.weights)

classOneFracTrain = OHETrainData.map(lambda data: data.label).mean()

logLossTrBase = OHETrainData.map(lambda data: computeLogLoss(classOneFracTrain, data.label) ).mean()
print 'Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase)

trainingPredictions = OHETrainData.map(lambda data: getP(data.features, model0.weights, model0.intercept))

logLossTrLR0 = evaluateResults(model0, OHETrainData)
print ('OHE Features Train Logloss:\tBaseline = {0:.3f}\tLogReg = {1:.3f}'.format(logLossTrBase, logLossTrLR0))


logLossValBase = OHEValidationData.map(lambda data: computeLogLoss(classOneFracTrain, data.label)).mean()

logLossValLR0 = evaluateResults(model0, OHEValidationData)
print ('OHE Features Validation Logloss:\tBaseline = {0:.3f}\tLogReg = {1:.3f}'.format(logLossValBase, logLossValLR0))

#------------------------
# Hash Functioning
#------------------------
sampleOne = [(0, 'mouse'), (1, 'black')]
sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]

# Use four buckets
sampOneFourBuckets = hashFunction(4, sampleOne, True)
sampTwoFourBuckets = hashFunction(4, sampleTwo, True)
sampThreeFourBuckets = hashFunction(4, sampleThree, True)

# Use one hundred buckets
sampOneHundredBuckets = hashFunction(100, sampleOne, True)
sampTwoHundredBuckets = hashFunction(100, sampleTwo, True)
sampThreeHundredBuckets = hashFunction(100, sampleThree, True)

print '\t\t 4 Buckets \t\t\t 100 Buckets'
print 'SampleOne:\t {0}\t\t {1}'.format(sampOneFourBuckets, sampOneHundredBuckets)
print 'SampleTwo:\t {0}\t\t {1}'.format(sampTwoFourBuckets, sampTwoHundredBuckets)
print 'SampleThree:\t {0}\t {1}'.format(sampThreeFourBuckets, sampThreeHundredBuckets)

numBucketsCTR = 2 ** 15
hashTrainData = rawTrainData.map(lambda data: parseHashPoint(data, numBucketsCTR))
hashTrainData.cache()
hashValidationData = rawValidationData.map(lambda data: parseHashPoint(data, numBucketsCTR))
hashValidationData.cache()
hashTestData = rawTestData.map(lambda data: parseHashPoint(data, numBucketsCTR))
hashTestData.cache()

print hashTrainData.take(1)

averageSparsityHash = computeSparsity(hashTrainData, numBucketsCTR, nTrain)
averageSparsityOHE = computeSparsity(OHETrainData, numCtrOHEFeats, nTrain)

print 'Average OHE Sparsity: {0:.10e}'.format(averageSparsityOHE)
print 'Average Hash Sparsity: {0:.10e}'.format(averageSparsityHash)

#------------------
# Test Code
#------------------

# TEST Loading and splitting the data (3a)
Test.assertTrue(all([rawTrainData.is_cached, rawValidationData.is_cached, rawTestData.is_cached]),'you must cache the split data')
Test.assertEquals(nTrain, 79911, 'incorrect value for nTrain')
Test.assertEquals(nVal, 10075, 'incorrect value for nVal')
Test.assertEquals(nTest, 10014, 'incorrect value for nTest')

# TEST Extract features (3b)
Test.assertEquals(numCategories[2][1], 855, 'incorrect implementation of parsePoint')
Test.assertEquals(numCategories[32][1], 4, 'incorrect implementation of parsePoint')

# TEST Create an OHE dictionary from the dataset (3c)
Test.assertEquals(numCtrOHEFeats, 233286, 'incorrect number of features in ctrOHEDict')
Test.assertTrue((0, '') in ctrOHEDict, 'incorrect features in ctrOHEDict')

# TEST Apply OHE to the dataset (3d)
numNZ = sum(parsedTrainFeat.map(lambda x: len(x)).take(5))
numNZAlt = sum(OHETrainData.map(lambda lp: len(lp.features.indices)).take(5))
Test.assertEquals(numNZ, numNZAlt, 'incorrect implementation of parseOHEPoint')
Test.assertTrue(withOneHot, 'oneHotEncoding not present in parseOHEPoint')

# TEST Handling unseen features (3e)
numNZVal = (OHEValidationData.map(lambda lp: len(lp.features.indices)).sum())
Test.assertEquals(numNZVal, 372080, 'incorrect number of features')

# TEST Logistic regression (4a)
Test.assertTrue(np.allclose(model0.intercept,  0.56455084025), 'incorrect value for model0.intercept')
Test.assertTrue(np.allclose(sortedWeights[0:5],[-0.45899236853575609, -0.37973707648623956, -0.36996558266753304,-0.36934962879928263, -0.32697945415010637]), 'incorrect value for model0.weights')

# TEST Log loss (4b)
Test.assertTrue(np.allclose([computeLogLoss(.5, 1), computeLogLoss(.01, 0), computeLogLoss(.01, 1)],[0.69314718056, 0.0100503358535, 4.60517018599]),'computeLogLoss is not correct')
Test.assertTrue(np.allclose([computeLogLoss(0, 1), computeLogLoss(1, 1), computeLogLoss(1, 0)],[25.3284360229, 1.00000008275e-11, 25.3284360229]),'computeLogLoss needs to bound p away from 0 and 1 by epsilon')

# TEST Baseline log loss (4c)
Test.assertTrue(np.allclose(classOneFracTrain, 0.22717773523), 'incorrect value for classOneFracTrain')
Test.assertTrue(np.allclose(logLossTrBase, 0.535844), 'incorrect value for logLossTrBase')

# TEST Predicted probability (4d)
Test.assertTrue(np.allclose(trainingPredictions.sum(), 18135.4834348),'incorrect value for trainingPredictions')

# TEST Evaluate the model (4e)
Test.assertTrue(np.allclose(logLossTrLR0, 0.456903), 'incorrect value for logLossTrLR0')

# TEST Validation log loss (4f)
Test.assertTrue(np.allclose(logLossValBase, 0.527603), 'incorrect value for logLossValBase')
Test.assertTrue(np.allclose(logLossValLR0, 0.456957), 'incorrect value for logLossValLR0')

# TEST Hash function (5a)
Test.assertEquals(sampOneFourBuckets, {2: 1.0, 3: 1.0}, 'incorrect value for sampOneFourBuckets')
Test.assertEquals(sampThreeHundredBuckets, {72: 1.0, 5: 1.0, 14: 1.0},'incorrect value for sampThreeHundredBuckets')

# TEST Creating hashed features (5b)
hashTrainDataFeatureSum = sum(hashTrainData.map(lambda lp: len(lp.features.indices)).take(20))
hashTrainDataLabelSum = sum(hashTrainData.map(lambda lp: lp.label).take(100))
hashValidationDataFeatureSum = sum(hashValidationData.map(lambda lp: len(lp.features.indices)).take(20))
hashValidationDataLabelSum = sum(hashValidationData.map(lambda lp: lp.label).take(100))
hashTestDataFeatureSum = sum(hashTestData.map(lambda lp: len(lp.features.indices)).take(20))
hashTestDataLabelSum = sum(hashTestData.map(lambda lp: lp.label).take(100))

Test.assertEquals(hashTrainDataFeatureSum, 772, 'incorrect number of features in hashTrainData')
Test.assertEquals(hashTrainDataLabelSum, 24.0, 'incorrect labels in hashTrainData')
Test.assertEquals(hashValidationDataFeatureSum, 776,'incorrect number of features in hashValidationData')
Test.assertEquals(hashValidationDataLabelSum, 16.0, 'incorrect labels in hashValidationData')
Test.assertEquals(hashTestDataFeatureSum, 774, 'incorrect number of features in hashTestData')
Test.assertEquals(hashTestDataLabelSum, 23.0, 'incorrect labels in hashTestData')

# TEST Sparsity (5c)
Test.assertTrue(np.allclose(averageSparsityOHE, 1.6717677e-04),'incorrect value for averageSparsityOHE')
Test.assertTrue(np.allclose(averageSparsityHash, 1.1805561e-03),'incorrect value for averageSparsityHash')
