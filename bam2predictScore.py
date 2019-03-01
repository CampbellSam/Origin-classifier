#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Nick Dickens'

# Simple Python script to generate wiggle files for kmer counts
# Nick Dickens, April 2015

import pysam
import sys
import os.path
import sklearn
import argparse


from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from numpy import median

parser = argparse.ArgumentParser(description='Generates a wiggle file from two bam files with the ration of ')
parser.add_argument('--chrNo', required=False, type=int, help='the chromosome index in the list to do the wig on (1-based)')
args = parser.parse_args()

chromosomeIndex = 0
if args.chrNo:
    chromosomeIndex = args.chrNo


# TO DO: Update this to take user input
# fileE = "/Users/nd48m/Google_Drive/Data/Origins/Leishmania/major/LmjFcombi.eS.bam"
baseDir = '/wtcmpbix/nd48m'
#baseDir = '/Users/nd48m'
bamFile = baseDir + '/Google_Drive/Data/Origins/Leishmania/major/LmjFcombi.g2.bam'

bedFileClass1 = baseDir + '/Google_Drive/Projects/2015/05_May/Machine_Learning/LmjF_origin_locations.bed'
bedFileClass2 = baseDir + '/Google_Drive/Projects/2015/05_May/Machine_Learning/LmjF_nonorigin_locations.bed'

# TO DO: And this
windowSize = 2500
kmerSize = 10

dataDir = baseDir + "/Google_Drive/Data/Origins_Training_Data"
# connect to the bam files
# TO DO: put a Try/Except around this
samfile = pysam.Samfile(bamFile, "rb")


'''
#create all kmers as SLIDING WINDOW along the read
def kmerize(dna, k):
    thisString = ""
    for i in range(0, len(dna)+1-k, 1):
        start = max(0, i)
        stop = min(len(dna), i + k) + 1
        thisString = thisString + dna[start:stop] + " "
    return thisString


'''
#create all kmers as TILES along the read
def kmerize(dna, k):
    thisString = ""
    for i in range(0, len(dna)+1-k, k):
        start = max(0, i)
        stop = min(len(dna), i + k) + 1
        thisString = thisString + dna[start:stop] + " "
    return thisString


# check the file is indexed
# TO DO: Call the samtools indexing pysam function if it isn't indexed
if not os.path.exists(bamFile + ".bai"):
    sys.exit('ERROR: There is no index for ' + bamFile + '!')


# do original classification

#categories = ['Origins', 'NonOrigins']


trainingData = sklearn.datasets.load_files(dataDir,random_state=42)
sys.stderr.write("Alpha\n")
# print (trainingData.data)


# Vectorize
#thisVect = CountVectorizer(ngram_range=(1, 1), max_df=0.5)
thisVect = CountVectorizer(ngram_range=(1, 1))
sys.stderr.write("Bravo\n")

myCounts = thisVect.fit_transform(trainingData.data)
sys.stderr.write("Charlie\n")

# Transform
thisTransformer = TfidfTransformer(norm='l2', use_idf=True).fit(myCounts)
myTransformed = thisTransformer.transform(myCounts)

# Classify
thisClf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-6, n_iter=10).fit(myTransformed, trainingData.target)

sys.stderr.write("Delta\n")



chromosomeList =  []
if chromosomeIndex:
    chromosomeList.append(samfile.references[chromosomeIndex-1])
else:
    chromosomeList = samfile.references

# iterate through each chromosome
for chromosome in chromosomeList:
    # get the length of the chromosome
    print("fixedStep  chrom=%s  start=1  step=%d  span=%d" % (chromosome, windowSize, windowSize))
    chromosomeLength = samfile.lengths[samfile.references.index(chromosome)]
    # read along the length of each chromosome
    for windowStart in range(0, chromosomeLength - windowSize, windowSize):
        kmerCount = 0
        alignmentsIterator = samfile.fetch(chromosome, windowStart, windowStart + windowSize)
        scoreList = []

        for alignment in alignmentsIterator:
            kmerizedList = []
            testData = kmerize(alignment.query_sequence,kmerSize)
            kmerizedList.append(testData)

            testVect = thisVect.transform(kmerizedList)
            testTransformed = thisTransformer.transform(testVect)

            score=thisClf.decision_function(testTransformed)
            scoreList.append(score)

        # generate ratio using corrected counts
        #		kmerPerKb = 1000*kmerCount/windowSize
        #		print "%4.3f" % kmerPerKb
        #		print "%4.3f" % kmerCount
        print("%4.3f" % median(scoreList))
samfile.close()
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
