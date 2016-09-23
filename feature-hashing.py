labVersion = 'cs190.1x-lab4-1.0.4'

from test_helper import Test
import numpy as np
from pyspark.mllib.linalg import SparseVector

#----------------------------
# Function Code 
#----------------------------

def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
    """Produce a one-hot-encoding from a list of features and an OHE dictionary.

    Note:
        You should ensure that the indices used to create a SparseVector are sorted.

    Args:
        rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
            feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
        OHEDict (dict): A mapping of (featureID, value) to unique integer.
        numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
            value).

    Returns:
        SparseVector: A SparseVector of length numOHEFeats with indices equal to the unique
            identifiers for the (featureID, value) combinations that occur in the observation and
            with values equal to 1.0.
    """

    uniqueFeatID=[]

    for item in rawFeats:
    	uniqueFeatID.append( OHEDict.get(item) )

    return SparseVector(numOHEFeats, sorted(uniqueFeatID), np.ones( len(uniqueFeatID) ))

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
	


#----------------------------
# Main Code 
#----------------------------

sampleOne = [(0, 'mouse'), (1, 'black')]
sampleTwo = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
sampleThree =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]
sampleDataRDD = sc.parallelize([sampleOne, sampleTwo, sampleThree])

sampleOHEDictManual = {}
sampleOHEDictManual[(0,'bear')] = 0
sampleOHEDictManual[(0,'cat')] = 1
sampleOHEDictManual[(0,'mouse')] = 2
sampleOHEDictManual[(1, 'black')] = 3
sampleOHEDictManual[(1, 'tabby')] = 4
sampleOHEDictManual[(2, 'mouse')] = 5
sampleOHEDictManual[(2, 'salmon')] = 6

aDense = np.array([0., 3., 0., 4.])
aSparse = SparseVector(4, [1,3],[3.,4.])

bDense = np.array([0., 0., 0., 1.])
bSparse = SparseVector(4, [3],[1.])

w = np.array([0.4, 3.1, -1.4, -.5])

sampleOneOHEFeatManual = SparseVector(7, [2,3], [1,1])
sampleTwoOHEFeatManual = SparseVector(7, [1,4,5],[1,1,1])
sampleThreeOHEFeatManual = SparseVector(7, [0,3,6],[1,1,1])

# Calculate the number of features in sampleOHEDictManual
numSampleOHEFeats = len(sampleOHEDictManual)

# Run oneHotEnoding on sampleOne
sampleOneOHEFeat = oneHotEncoding(sampleOne, sampleOHEDictManual, numSampleOHEFeats)

print sampleOneOHEFeat

sampleOHEData = sampleDataRDD.map(lambda data: oneHotEncoding(data, sampleOHEDictManual, numSampleOHEFeats ) )
print sampleOHEData.collect()

sampleDistinctFeats = (sampleDataRDD.flatMap(lambda data: data).distinct())

sampleOHEDict = sampleDistinctFeats.zipWithIndex().collectAsMap() #returns key-value pair as dictionary
print sampleOHEDict, "<-- sampleOHEDict"

sampleOHEDictAuto = createOneHotDict( sampleDataRDD )
print sampleOHEDictAuto, "<-- sampleOHEDictAuto from function"

#-----------------------------
# Testing Portion
#-----------------------------

Test.assertEqualsHashed(sampleOHEDictManual[(0,'bear')], 'b6589fc6ab0dc82cf12099d1c2d40ab994e8410c', "incorrect value for sampleOHEDictManual[(0,'bear')]")
Test.assertEqualsHashed(sampleOHEDictManual[(0,'cat')], '356a192b7913b04c54574d18c28d46e6395428ab', "incorrect value for sampleOHEDictManual[(0,'cat')]")
Test.assertEqualsHashed(sampleOHEDictManual[(0,'mouse')], 'da4b9237bacccdf19c0760cab7aec4a8359010b0', "incorrect value for sampleOHEDictManual[(0,'mouse')]")
Test.assertEqualsHashed(sampleOHEDictManual[(1,'black')], '77de68daecd823babbb58edb1c8e14d7106e83bb', "incorrect value for sampleOHEDictManual[(1,'black')]")
Test.assertEqualsHashed(sampleOHEDictManual[(1,'tabby')], '1b6453892473a467d07372d45eb05abc2031647a', "incorrect value for sampleOHEDictManual[(1,'tabby')]")
Test.assertEqualsHashed(sampleOHEDictManual[(2,'mouse')], 'ac3478d69a3c81fa62e60f5c3696165a4e5e6ac4', "incorrect value for sampleOHEDictManual[(2,'mouse')]")
Test.assertEqualsHashed(sampleOHEDictManual[(2,'salmon')], 'c1dfd96eea8cc2b62785275bca38ac261256e278', "incorrect value for sampleOHEDictManual[(2,'salmon')]")
Test.assertEquals(len(sampleOHEDictManual.keys()), 7,'incorrect number of keys in sampleOHEDictManual')
# TEST Sparse Vectors (1b)
Test.assertTrue(isinstance(aSparse, SparseVector), 'aSparse needs to be an instance of SparseVector')
Test.assertTrue(isinstance(bSparse, SparseVector), 'aSparse needs to be an instance of SparseVector')
Test.assertTrue(aDense.dot(w) == aSparse.dot(w), 'dot product of aDense and w should equal dot product of aSparse and w')
Test.assertTrue(bDense.dot(w) == bSparse.dot(w), 'dot product of bDense and w should equal dot product of bSparse and w')
# TEST OHE Features as sparse vectors (1c)
Test.assertTrue(isinstance(sampleOneOHEFeatManual, SparseVector), 'sampleOneOHEFeatManual needs to be a SparseVector')
Test.assertTrue(isinstance(sampleTwoOHEFeatManual, SparseVector), 'sampleTwoOHEFeatManual needs to be a SparseVector')
Test.assertTrue(isinstance(sampleThreeOHEFeatManual, SparseVector), 'sampleThreeOHEFeatManual needs to be a SparseVector')
Test.assertEqualsHashed(sampleOneOHEFeatManual, 'ecc00223d141b7bd0913d52377cee2cf5783abd6', 'incorrect value for sampleOneOHEFeatManual')
Test.assertEqualsHashed(sampleTwoOHEFeatManual, '26b023f4109e3b8ab32241938e2e9b9e9d62720a', 'incorrect value for sampleTwoOHEFeatManual')
Test.assertEqualsHashed(sampleThreeOHEFeatManual, 'c04134fd603ae115395b29dcabe9d0c66fbdc8a7', 'incorrect value for sampleThreeOHEFeatManual')
# TEST Define an OHE Function (1d)
Test.assertTrue(sampleOneOHEFeat == sampleOneOHEFeatManual, 'sampleOneOHEFeat should equal sampleOneOHEFeatManual')
Test.assertEquals(sampleOneOHEFeat, SparseVector(7, [2,3], [1.0,1.0]), 'incorrect value for sampleOneOHEFeat')
Test.assertEquals(oneHotEncoding([(1, 'black'), (0, 'mouse')], sampleOHEDictManual, numSampleOHEFeats), SparseVector(7, [2,3], [1.0,1.0]), 'incorrect definition for oneHotEncoding')

# TEST Apply OHE to a dataset (1e)
sampleOHEDataValues = sampleOHEData.collect()
Test.assertTrue(len(sampleOHEDataValues) == 3, 'sampleOHEData should have three elements')
Test.assertEquals(sampleOHEDataValues[0], SparseVector(7, {2: 1.0, 3: 1.0}),'incorrect OHE for first sample')
Test.assertEquals(sampleOHEDataValues[1], SparseVector(7, {1: 1.0, 4: 1.0, 5: 1.0}),'incorrect OHE for second sample')
Test.assertEquals(sampleOHEDataValues[2], SparseVector(7, {0: 1.0, 3: 1.0, 6: 1.0}),'incorrect OHE for third sample')

# TEST Pair RDD of (featureID, category) (2a)
Test.assertEquals(sorted(sampleDistinctFeats.collect()),[(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'),(1, 'tabby'), (2, 'mouse'), (2, 'salmon')],'incorrect value for sampleDistinctFeats')

# TEST OHE Dictionary from distinct features (2b)
Test.assertEquals(sorted(sampleOHEDict.keys()),[(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'),(1, 'tabby'), (2, 'mouse'), (2, 'salmon')],'sampleOHEDict has unexpected keys')
Test.assertEquals(sorted(sampleOHEDict.values()), range(7), 'sampleOHEDict has unexpected values')

# TEST Automated creation of an OHE dictionary (2c)
Test.assertEquals(sorted(sampleOHEDictAuto.keys()),[(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'),(1, 'tabby'), (2, 'mouse'), (2, 'salmon')],'sampleOHEDictAuto has unexpected keys')
Test.assertEquals(sorted(sampleOHEDictAuto.values()), range(7),'sampleOHEDictAuto has unexpected values')

