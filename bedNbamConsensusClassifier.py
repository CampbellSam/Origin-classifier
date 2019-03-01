#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Nick Dickens'

# Python 2 script to load training data from a bam file and bed files for classes
# then test this on a second bam file with a bed file of regions to classify
# does not write a training set to disc...but can optionally write the model
# to a file
# Nick Dickens, August 2015

import pysam
import sys
import os.path
import pandas as pd
import argparse
import random
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from Bio import SeqIO


parser = argparse.ArgumentParser(description='Classify read data, train from bam files and bed files a wiggle files and apply to a test set')
parser.add_argument('--trainFasta', required=False, type=str, help='a fasta file of training data')
parser.add_argument('--classBed', required=False, type=str, help='a bed file of training locations (should be specified at least twice)', action='append')
parser.add_argument('--machineIn', required=False, type=str, help='read a machine from this file (trainBam and classBed not required')
parser.add_argument('--machineOut', required=False, type=str, help='once trained write the machine to this file')
parser.add_argument('--testFasta', required=False, type=str, help='a bam file of test data')
parser.add_argument('--testBed', required=False, type=str, help='a bed file of locations to test')
parser.add_argument('--baseDir', required=False, type=str, help='common base directory for all training data (e.g. /Users/username/Data)')
parser.add_argument('--kmerSize', required=False, type=int, help='specify the kmer size (default is 10)')
parser.add_argument('--randomSeed', required=False, type=int, help='specify the random seed for randomizing the data set (default is none)')
parser.add_argument('--cpu', required=False, type=int, help='give the number of cpus to use in the classification (default is 1)')
parser.add_argument('--top10', required=False, action='store_true', help='print the top10 most informative features for the classifications')
args = parser.parse_args()

if args.trainFasta is True and args.classBed is True and args.machineIn is True:
    parser.error("--trainFasta + --classBed and --machineIn are mutually exclusive")
elif args.machineIn is None and (args.classBed is None or len(args.classBed)<2):
    parser.error("you need at least two --classBed classes to train the machine")
elif args.trainFasta is True and args.classBed is None:
    parser.error("--trainFasta needs at least two --classBed files to train on")
elif args.trainFasta is None and args.classBed is True:
    parser.error("--classBed needs at least one --trainFasta file to train on")
elif args.top10 is None and args.machineOut is None and (args.testBed is None or args.testFasta is None):
    parser.error("you have not specified a machineOut and one of your testBed/testFasta arguments is missing so there is nothing to run")

if args.machineIn is True and args.machineOut is True:
    parser.error("you can't read and write a machine in the same run (the machine won't be updated)")

if args.testFasta is True and args.testBed is None:
    parser.error("you have a test fasta without a test bed, you need both or none")
elif args.testFasta is None and args.testBed is True:
    parser.error("you have a test fasta without a test bed, you need both or none")


numCpu=1
if args.cpu is not None:
    numCpu = args.cpu



baseDir = ''

if args.baseDir:
    baseDir = args.baseDir
    baseDir = baseDir.rstrip('/')
    baseDir = baseDir + '/'


trainingFasta = ''
if args.trainFasta:
    trainingFasta = baseDir + args.trainFasta
trainingBedfiles = []
classesList = []

if args.machineIn is None:
    for bed in args.classBed:
        trainingBedfiles.append(bed)
        label = bed.split('/')[-1]
        classesList.append(label)

# set up other parameters
kmerSize = 10
if args.kmerSize:
    kmerSize = args.kmerSize


if args.randomSeed:
    random.seed(args.randomSeed)





# create all kmers as SLIDING WINDOW along the sequence
# because each sequence is only a single observation, didn't need
# to do this with reads (as they were lots of observations with
# lots of starting points
def kmerize(dna, k):
    thisString = ""
    for i in range(0, len(dna)+1-k, 1):
        start = max(0, i)
        stop = min(len(dna), i + k) + 1
        thisString = thisString + dna[start:stop] + " "
    return thisString


def kmerizeFasta(sequence, k):
    kmerData = kmerize(sequence,k)
    return kmerData


#if machine is specified load it and do not train
trainingData = []
vectorizer = None
data_vectors = None
transformer = None
classifier = None
target_labels = []



#checkif there is something to load instead
if args.machineIn:
    prefix = args.machineIn

    sys.stderr.write("Loading vectorizer...")
    vectorizer = joblib.load(prefix + ".vectorizer")
    kmerSize = joblib.load(prefix + ".kmerSize")
    sys.stderr.write("done\n")

    sys.stderr.write("Loading transformer...")
    transformer = joblib.load(prefix + ".transformer")
    sys.stderr.write("done\n")

    sys.stderr.write("Loading vectors...")
    data_vectors = joblib.load(prefix + ".data_vectors")
    sys.stderr.write("done\n")

    sys.stderr.write("Loading target_labels and names...")
    target_labels = joblib.load(prefix + ".target_labels")
    classesList = joblib.load(prefix + ".classes")
    sys.stderr.write("done\n")

    sys.stderr.write("Loading classifier...")
    classifier = joblib.load(prefix + ".classifier")
    sys.stderr.write("done\n")

    if os.path.exists(prefix + ".randomSeed"):
        randomSeed = joblib.load(prefix + ".randomSeed")
        random.seed(randomSeed)

else:
    # do original classification
    trainingFastaDict = {}
    #load the fasta
    handle = open(baseDir + trainingFasta, "rU")
    trainingFastaDict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    handle.close()
    #open training bed files
    for thisClass in classesList:
        bedFile = baseDir + trainingBedfiles[classesList.index(thisClass)] # take the file with the same index as the class
        bedTable = pd.read_table(bedFile,sep='\t',header=None,names=['chr','start','end'])
        for index, bedRecord in bedTable.iterrows() :
            thisStart = bedRecord['start']
            thisEnd = bedRecord['end']
            if bedRecord['end']<bedRecord['start']:
                sys.stderr.write('Warning: end < start so switching records in file %s start: %d end: %d\n' % (thisClass, bedRecord['start'],bedRecord['end']) )
                thisEnd = bedRecord['start']
                thisStart = bedRecord['end']

            seqPart = trainingFastaDict[bedRecord['chr']][thisStart:thisEnd]
            trainingData.append(kmerizeFasta(seqPart.seq, kmerSize))
            target_labels.append(classesList.index(thisClass))

        #end of the for loop for classesList

    # Randomize the data, so both lists are in the same order but not in the original order
    newIndices = range(0,len(target_labels),1)
    random.shuffle(newIndices)

    target_labels = [ target_labels[i] for i in newIndices]
    trainingData = [ trainingData[i] for i in newIndices]
    #sys.stderr.write("Charlie\n")


    # Vectorize
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    data_vectors = vectorizer.fit_transform(trainingData)
    #sys.stderr.write("Delta\n")

    # Transform
    transformer = TfidfTransformer(norm='l2', use_idf=True).fit(data_vectors)
    transformed_data = transformer.transform(data_vectors)

    #sys.stderr.write("Echo\n")
    # Classify
    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-6, n_iter=10, n_jobs=numCpu).fit(transformed_data, target_labels)
    selfPredicted = classifier.predict(transformed_data)
    acc = np.mean(selfPredicted == target_labels)

    print("Self accuracy: %3.2f %%" % (acc * 100))
    #end of the else

    #sys.stderr.write("Foxtrot\n") # classifier loaded, etc

    #write out the machine if asked to do so
if args.machineOut:
    prefix = args.machineOut

    sys.stderr.write("Dumping vectorizer...")
    joblib.dump(vectorizer, prefix + ".vectorizer")
    joblib.dump(kmerSize, prefix + ".kmerSize")
    sys.stderr.write("done\n")

    sys.stderr.write("Dumping transformer...")
    joblib.dump(transformer, prefix + ".transformer")
    sys.stderr.write("done\n")

    sys.stderr.write("Dumping vectors...")
    joblib.dump(data_vectors, prefix + ".data_vectors")
    sys.stderr.write("done\n")

    sys.stderr.write("Dumping target_labels and classes...")
    joblib.dump(target_labels, prefix + ".target_labels")
    joblib.dump(classesList, prefix + ".classes")
    sys.stderr.write("done\n")

    sys.stderr.write("Dumping classifier...")
    joblib.dump(classifier, prefix + ".classifier")
    sys.stderr.write("done\n")

    if args.randomSeed:
        sys.stderr.write("Dumping random seed...")
        joblib.dump(args.randomsSeed, prefix + ".randomSeed")
        sys.stderr.write("done\n")

#sys.stderr.write("Golf\n") # classifier loaded, etc


#if test data
if args.testFasta:
    testFasta = args.testFasta
    #load the sequenec file
    handle = open(baseDir + testFasta, "rU")
    testFastaDict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    handle.close()

    testBedfile = baseDir + args.testBed
    testBedTable = pd.read_table(testBedfile,sep='\t',header=None,names=['chr','start','end'])
    testData = []
    testLabels = [] # debugging only
    resultsTable = []
    for index, bedRecord in testBedTable.iterrows() :
        #location = str(bedRecord['chr']) + ":" + str(bedRecord['start']) + "-" + str(bedRecord['end'])
        thisStart = bedRecord['start']
        thisEnd = bedRecord['end']
        if bedRecord['end']<bedRecord['start']:
            sys.stderr.write('Warning: end < start so switching records in file %s start: %d end: %d\n' % (testBedfile, bedRecord['start'],bedRecord['end']) )
            thisEnd = bedRecord['start']
            thisStart = bedRecord['end']
        thisLocation = str(bedRecord['chr']) + ":" + str(thisStart) + "-" + str(thisEnd)
        resultsTable.append(thisLocation)

        seqPart = testFastaDict[bedRecord['chr']][thisStart:thisEnd]
        #print(seqPart.seq)
        testData.append(kmerizeFasta(str(seqPart.seq), kmerSize))
        #print(testData)
        testLabels.append(0) # debugging only
        #TO DO add the ability to write a bunch of predictions to a file and test against this


    #sys.stderr.write("Hotel\n")
    test_vectors = vectorizer.transform(testData)
    test_transformed_data = transformer.transform(test_vectors)
    #sys.stderr.write("India\n")
    #score=thisClf.decision_function(testTransformed)
    test_predictions = classifier.predict(test_transformed_data)
    testAcc = np.mean(test_predictions == testLabels)

    print("Test accuracy (assuming all are class %s): %3.2f %%" % (classesList[0], testAcc * 100))

    print ("Location\tPrediction")
    for i in range(0,len(test_predictions)):
        print ("%s\t%s" % (resultsTable[i],classesList[test_predictions[i]]))
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    #stolen from http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html
    return s if len(s) <= 80 else s[:77] + "..."

if args.top10 is True:
    feature_names = np.asarray(vectorizer.get_feature_names())
    print("top 10 keywords per class:")
    for i, category in enumerate(classesList):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print(top10)
        print("%s: %s" % (category, "\n".join(feature_names[top10])))