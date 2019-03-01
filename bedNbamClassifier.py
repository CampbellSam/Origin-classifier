#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Nick Dickens'

# Python 2 script to load training data from a bam file and bed files for classes
# then test this on a second bam file with a bed file of regions to classify
# does not write a training set to disc...but can optionally write the model
# to a file
# The plan is to replace this with python3 and object-oriented version
# Nick Dickens, July 2015

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


def readCommandlineOptions():
    parser = argparse.ArgumentParser(description='Classify read data, train from bam files and bed files a wiggle files and apply to a test set')
    parser.add_argument('--trainBam', required=False, type=str, help='a bam file of training data (can be a multiple times)', action='append')
    parser.add_argument('--classBed', required=False, type=str, help='a bed file of training locations (should be specified at least twice)', action='append')
    parser.add_argument('--machineIn', required=False, type=str, help='read a machine from this file (trainBam and classBed not required')
    parser.add_argument('--machineOut', required=False, type=str, help='once trained write the machine to this file')
    parser.add_argument('--testBam', required=False, type=str, help='a bam file of test data')
    parser.add_argument('--testBed', required=False, type=str, help='a bed file of locations to test')
    parser.add_argument('--baseDir', required=False, type=str, help='common base directory for all training data (e.g. /Users/username/Data)')
    parser.add_argument('--kmerSize', required=False, type=int, help='specify the kmer size (default is 10)')
    parser.add_argument('--randomSeed', required=False, type=int, help='specify the random seed for randomizing the data set (default is none)')
    parser.add_argument('--cpu', required=False, type=int, help='give the number of cpus to use in the classification (default is 1)')
    parser.add_argument('--top10', required=False, action='store_true', help='print the top10 most informative features for the classifications')
    args = parser.parse_args()

    if args.trainBam is True and args.classBed is True and args.machineIn is True:
        parser.error("--trainBam + --classBed and --machineIn are mutually exclusive")
    elif args.machineIn is None and (args.classBed is None or len(args.classBed)<2):
        parser.error("you need at least two --classBed classes to train the machine")
    elif args.trainBam is True and args.classBed is None:
        parser.error("--trainBam needs at least two --classBed files to train on")
    elif args.trainBam is None and args.classBed is True:
        parser.error("--classBed needs at least one --trainBam file to train on")
    elif args.top10 is None and args.machineOut is None and (args.testBed is None or args.testBam is None):
        parser.error("you have not specified a machineOut and one of your testBed/testBam arguments is missing so there is nothing to run")

    if args.machineIn is True and args.machineOut is True:
        parser.error("you can't read and write a machine in the same run (the machine won't be updated)")

    if args.testBam is True and args.testBed is None:
        parser.error("you have a test bam without a test bed, you need both or none")
    elif args.testBam is None and args.testBed is True:
        parser.error("you have a test bam without a test bed, you need both or none")
    return args






# Functions

def kmerize(dna, k):
    """ create all kmers as TILES along the dna read in a format suitable for scikit-learn vectorizer
    :param dna: a string of the dna read sequence
    :param k: the size of kmer
    :return: thisString: a string of the kmers as words separated by space
    """
    thisString = ""
    for i in range(0, len(dna)+1-k, k):
        start = max(0, i)
        stop = min(len(dna), i + k) + 1
        thisString = thisString + dna[start:stop] + " "
    return thisString

def kmerizeAlignments(alignmentsIterator, k):
    """ Provides text input that is suitable for the scikit-learn vectorizer
    :param alignmentsIterator: a pysam alignments iterator for the region
    :param k: the size of the kmer tiles
    :return: kmerizedText: a string of kmer words and each read from the alignments iterator makes a paragraph.
    """
    kmerizedText = ""
    for alignment in alignmentsIterator:
        kmerData = kmerize(alignment.query_sequence,k)
        kmerizedText = kmerizedText + kmerData + '\n'
    return kmerizedText

def trim(thisString):
    """ Trim a string to fit an 80-column display
    :param thisString:
    :return: this String as it is or truncated to 80 characters (that is 77 followed by ...)
    """
    if len(thisString) <= 80:
        return thisString
    else:
        return thisString[:77] + "..."


# Classes
class Machine(object):
    """ A machine object that can handle all of the necessary parts of the scikit-learn objects to be able to persist between
    sessions.  The data are stored as numpy arrays using the sklearn.externals joblib module.
    """
    def __init__(self, machineName,vectorizer=None, kmerSize=None, transformer=None, dataVectors=[], dataLabels=[], classList=[], classifier=None, randomSeed=None):
        """ initialize the machine, really just check the machineName is defined and not empty
        :param machineName:
        :param vectorizer:
        :param kmerSize:
        :param transformer:
        :param dataVectors:
        :param dataLabels:
        :param classList:
        :param classifier:
        :param randomSeed:
        :return: Machine object
        """
        if machineName is None or machineName == "":
            self.machineName = machineName
            sys.exit("ERROR: the machineName is empty so cannot create a new machine object!\n")

        self.machineName = machineName


    def checkMachineFiles(self):
        """ checks that all the required files exist
        :return:
        """
        extensionList = ("vectorizer", "kmerSize", "transformer", "target_labels", "classes", "classifier")
        for extension in extensionList:
            if not os.path.exists(self.machineName + "." + extension):
                sys.exit("ERROR: the machine file %s.%s does not exist!\n" % (self.machineName, extension))




    def load(self):
        """ load a saved machine from existing files
        if all goes well the machine attributes will be populated
        :return:
        """

        sys.stderr.write("Loading machine (" + self.machineName + ") ...")

        self.checkMachineFiles()

        self.vectorizer = joblib.load(self.machineName + ".vectorizer")
        self.kmerSize = joblib.load(self.machineName + ".kmerSize")
        self.transformer = joblib.load(self.machineName + ".transformer")
        self.data_vectors = joblib.load(self.machineName + ".data_vectors")
        self.target_labels = joblib.load(self.machineName + ".target_labels")
        self.classesList = joblib.load(self.machineName + ".classes")
        self.classifier = joblib.load(self.machineName + ".classifier")

        # the randomseed is optional so it might not exist
        if os.path.exists(self.machineName + ".randomSeed"):
            self.randomSeed = joblib.load(self.machineName + ".randomSeed")
            random.seed(self.randomSeed)

        sys.stderr.write("done\n")


    def save(self):
        """ save a machine to a group of files for future use
        :return:
        """

        sys.stderr.write("Dumping vectorizer (" + self.machineName + ")...")
        joblib.dump(self.vectorizer, self.machineName + ".vectorizer")
        joblib.dump(self.kmerSize, self.machineName + ".kmerSize")
        joblib.dump(self.transformer, self.machineName + ".transformer")
        joblib.dump(self.data_vectors, self.machineName + ".data_vectors")
        joblib.dump(self.target_labels, self.machineName + ".target_labels")
        joblib.dump(self.classesList, self.machineName + ".classes")
        joblib.dump(self.classifier, self.machineName + ".classifier")

        if args.randomSeed:
            sys.stderr.write("Dumping random seed...")
            joblib.dump(self.randomSeed, self.machineName + ".randomSeed")
        sys.stderr.write("done\n")

        self.checkMachineFiles()






# Check if there is something to load instead
# if a machine is specified load it and do not train
if args.machineIn:
    pass
else:
# do the classification
#
    # Open and read training bed files
    for thisClass in classesList:
        #open the bam file
        for trainingBamfile in trainingBamfiles:
            thisBam = pysam.Samfile(baseDir + trainingBamfile, "rb")
            bedFile = baseDir + trainingBedfiles[classesList.index(thisClass)] # take the file with the same index as the class
            bedTable = pd.read_table(bedFile,sep='\t',header=None,names=['chr','start','end'])
            for index, bedRecord in bedTable.iterrows() :
                thisStart = bedRecord['start']
                thisEnd = bedRecord['end']
                if bedRecord['end']<bedRecord['start']:
                    sys.stderr.write('Warning: end < start so switching records in file %s start: %d end: %d\n' % (thisClass, bedRecord['start'],bedRecord['end']) )
                    thisEnd = bedRecord['start']
                    thisStart = bedRecord['end']

                alignmentsIterator = thisBam.fetch(bedRecord['chr'], thisStart, thisEnd + 1)
                trainingData.append(kmerizeAlignments(alignmentsIterator, kmerSize))
                target_labels.append(classesList.index(thisClass))
            thisBam.close()

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


    #Write out the machine if asked to do so
    #TO DO: make this default and set a default name, etc
if args.machineOut:
    pass

#If there is test data
if args.testBam:
    testBamfile = baseDir + args.testBam
    testBam = pysam.Samfile(testBamfile, "rb")
    testBedfile = baseDir + args.testBed
    testBedTable = pd.read_table(testBedfile,sep='\t',header=None,names=['chr','start','end'])
    testData = []
    testLabels = [] # debugging only
    resultsTable = []
    for index, bedRecord in testBedTable.iterrows() :
        thisStart = bedRecord['start']
        thisEnd = bedRecord['end']
        if bedRecord['end']<bedRecord['start']:
            sys.stderr.write('Warning: end < start so switching records in file %s start: %d end: %d\n' % (testBedfile, bedRecord['start'],bedRecord['end']) )
            thisEnd = bedRecord['start']
            thisStart = bedRecord['end']
        thisLocation = str(bedRecord['chr']) + ":" + str(thisStart) + "-" + str(thisEnd)
        resultsTable.append(thisLocation)
        alignmentsIterator = testBam.fetch(bedRecord['chr'], thisStart, thisEnd + 1)
        testData.append(kmerizeAlignments(alignmentsIterator, kmerSize))
        testLabels.append(0) # debugging only, default label is the first training class
    testBam.close()

    test_vectors = vectorizer.transform(testData)
    test_transformed_data = transformer.transform(test_vectors)
    #score=thisClf.decision_function(testTransformed)
    test_predictions = classifier.predict(test_transformed_data)
    testAcc = np.mean(test_predictions == testLabels)

    print("Test accuracy (assuming all are class %s): %3.2f %%" % (classesList[0], testAcc * 100))

    print ("Location\tPrediction")
    for i in range(0,len(test_predictions)):
        print ("%s\t%s" % (resultsTable[i],classesList[test_predictions[i]]))

if args.top10 is True:
    feature_names = np.asarray(vectorizer.get_feature_names())
    print("top 10 keywords per class:")
    for i, category in enumerate(classesList):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print(top10)
        print("%s: %s" % (category, "\n".join(feature_names[top10])))

if __name__ == "__main__":
    args = readCommandlineOptions()

    # set up defaults
    numCpu=1
    kmerSize = 10
    baseDir = ''
    trainingBamfiles = []
    trainingBedfiles = []

    # set up the data variables
    vectorizer = None
    data_vectors = None
    transformer = None
    classifier = None
    trainingData = []
    target_labels = []
    classesList = []



    # check the input arguments to update defaults
    if args.cpu is not None:
        numCpu = args.cpu

    if args.baseDir:
        baseDir = args.baseDir
        baseDir = baseDir.rstrip('/')

    if args.machineIn is None:
        for bam in args.trainBam:
            trainingBamfiles.append(bam)

    if args.machineIn is None:
        for bed in args.classBed:
            trainingBedfiles.append(bed)
            label = bed.split('/')[-1]
            classesList.append(label)

    if args.kmerSize:
        kmerSize = args.kmerSize

    if args.randomSeed:
        random.seed(args.randomSeed)

    # Check the bam file is indexed (otherwise access doesn't work)
    # TO DO: Call the samtools indexing pysam function if it isn't indexed
    # this requires samtools to be installed, etc but should be ok since
    # why would you install pysam and not samtools?!
    if args.trainBam is True:
        for bamFile in trainingBamfiles:
            if not os.path.exists(baseDir + bamFile + ".bai"):
                sys.exit('ERROR: There is no index for ' + baseDir + bamFile + '!')

    if args.testBam is True:
        if not os.path.exists(baseDir + args.testBam + ".bai"):
            sys.exit('ERROR: There is no index for ' + baseDir + args.testBam + '!')



