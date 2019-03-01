#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Nick Dickens'

# Python 2 script to load test data from a bam file and bed files for classes
# then test this on a second bam file with a bed file of regions to classify
# does not write a training set to disc...
# Nick Dickens, July 2015

import pysam
import sys
import os.path
import sklearn
import pandas as pd
import argparse
import random
import numpy as np

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from numpy import median

#parser = argparse.ArgumentParser(description='Generates a wiggle file from two bam files with the ration of ')
#parser.add_argument('--chrNo', required=False, type=int, help='the chromosome index in the list to do the wig on (1-based)')
#args = parser.parse_args()


# TO DO: Update these to take user input

baseDir = '/wtcmpbix/nd48m'
#baseDir = '/Users/nd48m'
#trainingBamfile = baseDir + '/Google_Drive/Data/Origins/Leishmania/major/LmjFcombi.eS.bam'
trainingBamfile = baseDir + '/Google_Drive/Data/Origins/Trypanosoma/brucei/MFAseq/TbrCombi.eS.bam'


#trainingBedDir= baseDir + '/Google_Drive/Projects/2015/05_May/Machine_Learning/'
#trainingBedClass1 = 'LmjF_origin_locations.bed'
#trainingBedClass2 = 'LmjF_nonorigin_locations.bed'
trainingBedDir= baseDir + '/Google_Drive/Data/BED/'
trainingBedClass1 = 'Tbru_SSR_origins.bed'
trainingBedClass2 = 'Tbru_SSR_nonorigins.bed'


classesList=[trainingBedClass1, trainingBedClass2]




#testBamfile = baseDir + '/Google_Drive/Data/Lbraziliensis/L.braziliensisM2904.bam'
#testBed = baseDir + '/Google_Drive/Data/BED/LmjF_origin_regions_in_LbrM.bed'
#testBamfile = baseDir + '/Google_Drive/Data/Origins/Leishmania/major/LmjFcombi.g2.bam'
#testBed = baseDir + '/Google_Drive/Projects/2015/05_May/Machine_Learning/LmjF_origin_locations.bed'
#testBamfile = baseDir + '/Google_Drive/Data/Origins/Leishmania/mexicana/L.mexicanaU1103G2.rmdup.bam.valid.bam'
#testBed = baseDir + '/Google_Drive/Data/BED/LmxM_origin_locations.bed'
testBamfile = baseDir + '/Google_Drive/Data/Origins/Trypanosoma/brucei/MFAseq/TbrCombi.g2.bam'
testBed = baseDir + '/Google_Drive/Data/BED/Tbru_SSR_origins.bed'



# TO DO: check the bam files are indexed
# check the file is indexed
for bamFile in [trainingBamfile, testBamfile]:
    if not os.path.exists(bamFile + ".bai"):
        sys.exit('ERROR: There is no index for ' + bamFile + '!')
# TO DO: Call the samtools indexing pysam function if it isn't indexed


# TO DO: And make these load from user input (can default to these values)
windowSize = 2500
kmerSize = 10
randomseed = 42



#create all kmers as TILES along the read
def kmerize(dna, k):
    thisString = ""
    for i in range(0, len(dna)+1-k, k):
        start = max(0, i)
        stop = min(len(dna), i + k) + 1
        thisString = thisString + dna[start:stop] + " "
    return thisString


def kmerizeAlignments(alignmentsIterator, k):
    #kmerizedList = []
    kmerizedList = ""
    for alignment in alignmentsIterator:
        kmerData = kmerize(alignment.query_sequence,k)
        kmerizedList = kmerizedList + kmerData + '\n'
    #    kmerizedList.append(kmerData)
    return kmerizedList



sys.stderr.write("Alpha\n")


# do original classification
trainingData = []
trainingLabels = []
#trainingData = sklearn.datasets.load_files(dataDir,random_state=randomseed)

#open training bed files
for thisClass in classesList:
    #open the bam file
    thisBam = pysam.Samfile(trainingBamfile, "rb")

    bedTable = pd.read_table(trainingBedDir + thisClass,sep='\t',header=None,names=['chr','start','end'])
    for index, bedRecord in bedTable.iterrows() :
        thisStart = bedRecord['start']
        thisEnd = bedRecord['end']
        if bedRecord['end']<bedRecord['start']:
            sys.stderr.write('Warning: end < start so switching records in file %s start: %d end: %d\n' % (thisClass, bedRecord['start'],bedRecord['end']) )
            thisEnd = bedRecord['start']
            thisStart = bedRecord['end']

        #location = str(bedRecord['chr']) + ":" + str(bedRecord['start']) + "-" + str(bedRecord['end'])
        alignmentsIterator = thisBam.fetch(bedRecord['chr'], thisStart, thisEnd + 1)
        trainingData.append(kmerizeAlignments(alignmentsIterator, kmerSize))
        trainingLabels.append(classesList.index(thisClass))
    thisBam.close()


sys.stderr.write("Bravo\n")

# Randomize the data, so both lists are in the same order but not in the original order
random.seed(randomseed)
newIndices = range(0,len(trainingLabels),1)
random.shuffle(newIndices)

trainingLabels = [ trainingLabels[i] for i in newIndices]
trainingData = [ trainingData[i] for i in newIndices]


# Vectorize
thisVect = CountVectorizer(ngram_range=(1, 1))
myCounts = thisVect.fit_transform(trainingData)
sys.stderr.write("Charlie\n")

# Transform
thisTransformer = TfidfTransformer(norm='l2', use_idf=False).fit(myCounts)
myTransformed = thisTransformer.transform(myCounts)

sys.stderr.write("Delta\n")
# Classify
thisClf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-6, n_iter=10).fit(myTransformed, trainingLabels)
sys.stderr.write("Echo\n")


selfPredicted = thisClf.predict(myTransformed)
acc = np.mean(selfPredicted == trainingLabels)

print("Self accuracy: %s" % acc)


# Load test regions
testBam = pysam.Samfile(testBamfile, "rb")

testBedTable = pd.read_table(testBed,sep='\t',header=None,names=['chr','start','end'])
testData = []
testLabels = [] # debudding only
for index, bedRecord in testBedTable.iterrows() :
    #location = str(bedRecord['chr']) + ":" + str(bedRecord['start']) + "-" + str(bedRecord['end'])
    alignmentsIterator = testBam.fetch(bedRecord['chr'], bedRecord['start'], bedRecord['end'] + 1)
    testData.append(kmerizeAlignments(alignmentsIterator, kmerSize))
    testLabels.append(0) # debugging only
testBam.close()

sys.stderr.write("Foxtrot\n")
testVect = thisVect.transform(testData)
testTransformed = thisTransformer.transform(testVect)
sys.stderr.write("Golf\n")
#score=thisClf.decision_function(testTransformed)
testPredictions = thisClf.predict(testTransformed)
testAcc = np.mean(testPredictions == testLabels)

print("Test accuracy: %s" % testAcc)

'''
The edits to make to the Gbrowse formatting for these tracks
glyph           = wiggle_xyplot
graph_type      = line
height          = 100
color           = black
bgcolor         = mediumblue
fgcolour        = mediumblue
linewidth       = 2
min_score       = 0
max_score       = <max score?>
'''
