#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Nick Dickens'

# Simple Python script to generate training data from reads and bed files
# Nick Dickens, July 2015

import pysam
import sys
import os
import collections
import re
import sklearn
import pandas as pd
import hashlib



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from numpy import median

# TO DO: Update this to take user input, until then:
baseDir = '/wtcmpbix/nd48m'
#baseDir = '/Users/nd48m'
bamFile = baseDir + '/Google_Drive/Data/Origins/Leishmania/major/LmjFcombi.eS.bam'


bedDir= baseDir + '/Google_Drive/Projects/2015/05_May/Machine_Learning/'
bedFileClass1 = 'LmjF_origin_locations.bed'
bedFileClass2 = 'LmjF_nonorigin_locations.bed'

classesList=[bedFileClass1, bedFileClass2]

outputDir = baseDir + '/Google_Drive/Data/Origins_Training_Data/'

windowSize = 2500
kmerSize = 10




################## RUNIT ##################

# check the bam file is indexed
# TO DO: Call the samtools indexing pysam function if it isn't indexed
if not os.path.exists(bamFile + ".bai"):
    sys.exit('ERROR: There is no index for ' + bamFile + '!')

# open the bam file
# TO DO: put a Try/Except around this
samfile = pysam.Samfile(bamFile, "rb")

'''
#create all kmers as SLIDING WINDOW along the read
def kmerize(dna, k):
    thisString = ""
    for i in range(0, len(dna)+1-k, 1):
        start = max(0, i - k)
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





for thisClass in classesList:
#thisClass=classesList[0]

    directory = outputDir + thisClass
    if not os.path.exists(directory):
        os.makedirs(directory)

    bedTable = pd.read_table(bedDir + thisClass,sep='\t',header=None,names=['chr','start','end'])

    readCounter = 0

    for index, bedRecord in bedTable.iterrows() :
       location = str(bedRecord['chr']) + ":" + str(bedRecord['start']) + "-" + str(bedRecord['end'])
       #print (location)
       alignmentsIterator = samfile.fetch(bedRecord['chr'], bedRecord['start'], bedRecord['end'] + 1)
       for alignment in alignmentsIterator:
           thisSample = kmerize(alignment.query_sequence,kmerSize)
           m = hashlib.md5()
           m.update(alignment.query_sequence)
           readId = m.hexdigest()
           text_file = open(directory + "/" + readId, "w")
           text_file.write(thisSample)
           text_file.close()
           readCounter += 1
       print (location + " had " + str(readCounter) + " reads")


