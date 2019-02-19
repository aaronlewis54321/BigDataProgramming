import pyspark
from operator import add
from pyspark.context import SparkContext
from pyspark import SparkConf

conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

def parseNeighbors(x):
    arr = []
    for i in range (2, len(x)):
        if i%2 == 0:
            arr.append((x[0], x[i]))
    return arr
def flatten(c):
    return [item for sublist in c for item in sublist]

def computeContribs(urls, rank):
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


# Load the adjacency list file
AdjList1 = sc.textFile("02AdjacencyList.txt")
print AdjList1.collect()

AdjList2 = AdjList1.map(lambda line : parseNeighbors(line))  # 1. Replace the lambda function with yours
AdjList3 = sc.parallelize(flatten(AdjList2.collect()))
AdjList3.persist()
print AdjList3.collect()

nNumOfNodes = AdjList2.count()
print "Total Number of nodes"
print nNumOfNodes

# Initialize each page's rank; since we use mapValues, the resulting RDD will have the same partitioner as links
print "Initialization"
link = AdjList3.groupByKey()
PageRankValues = link.map(lambda v : (v[0], 1.0)) 
#PageRankValues = links.mapValues(lambda v : (v[0], 1.0))  # 3. Replace the lambda function with yours
print PageRankValues.collect()

#Run 30 iterations
print "Run 30 Iterations"
for i in range(1, 50):
    print "Number of Iterations"
    print i
    JoinRDD = link.join(PageRankValues)
    print "join results"
    print JoinRDD.collect()
    contributions = JoinRDD.flatMap(lambda x : computeContribs(x[1][0], x[1][1]))  # 4. Replace the lambda function with yours
    print "contributions"
    print contributions.collect()
    adder = 0
    accumulations = contributions.reduceByKey(add).mapValues(lambda x: x*0.85 + 0.03)
    print "accumulations"
    print accumulations.collect()
    PageRankValues = accumulations.mapValues(lambda v : v)  # 6. Replace the lambda function with yours
    print "PageRankValues"
    print PageRankValues.collect()

print "=== Final PageRankValues ==="
print PageRankValues.collect()

# Write out the final ranks
#PageRankValues.coalesce(1).saveAsTextFile("../Assignment2/PageRankValues_Final.txt")

