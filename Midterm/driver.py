from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
import time
sc = SparkContext(appName="MNISTDigitsDT")

def parsePoint(line):
  #Parse a line of text into an MLlib LabeledPoint object
  values = line.split(',')
  values = [0 if e == '' else int(e) for e in values]
  return LabeledPoint(int(values[0]), values[1:])


fileNameTrain = 'train.csv'
fileNameTest = 'test.csv'
mnist_train = sc.textFile(fileNameTrain)
mnist_test = sc.textFile(fileNameTest)

#skip header
header = mnist_train.first() #extract header
mnist_train = mnist_train.filter(lambda x:x !=header)
#filter out header using a lambda
print mnist_train.first()

labeledPoints = mnist_train.map(parsePoint)
#Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = labeledPoints.randomSplit([0.7, 0.3])
print mnist_train.first()

depthLevel = 4
treeLevel = 3
#start timer
start_time = time.time()
#this is building a model using the Random Forest algorithm from Spark MLLib
model = RandomForest.trainClassifier(trainingData, numClasses=10,
  categoricalFeaturesInfo={},
  numTrees=treeLevel, featureSubsetStrategy="auto",
  impurity='gini', maxDepth=depthLevel, maxBins=32)
print("Training time --- %s seconds ---" % (time.time() - start_time))


# Evaluate model on test instances and compute test error
#start timer
start_time = time.time()
#make predictions using the Machine Learning created prior
predictions = model.predict(testData.map(lambda x: x.features))
#validate predictions using the training set
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print("Prediction time --- %s seconds ---" % (time.time() - start_time))
#print('Learned classification tree model:')
#print(model.toDebugString())

bestModel = None
bestTestErr = 100
#Define a range of hyperparameters to try
maxDepths = range(4,10)
maxTrees = range(3,10)

#Loop over parameters for depth and tree level(s)
for depthLevel in maxDepths:
 for treeLevel in maxTrees:

  #start timer
  start_time = time.time()
  #Train RandomForest machine learning classifier
  model = RandomForest.trainClassifier(trainingData,
    numClasses=10, categoricalFeaturesInfo={},
    numTrees=treeLevel, featureSubsetStrategy="auto",
    impurity='gini', maxDepth=depthLevel, maxBins=32)

  #Make predictions using the model created above
  predictions = model.predict(testData.map(lambda x: x.features))
  #Join predictions with actual values from the data and determine the error rate
  labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
  testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())

  #Print information about the model as we proceed with each iteration of the loop
  print ('\maxDepth = {0:.1f}, trees = {1:.1f}: trainErr = {2:.5f}'
         .format(depthLevel, treeLevel, testErr))
  print("Prediction time --- %s seconds ---" % (time.time() - start_time))
  if (testErr < bestTestErr):

      bestModel = model
      bestTestErr = testErr

print ('Best Test Error: = {0:.3f}\n'.format(bestTestErr))
