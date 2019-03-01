#Overview



#Features

#Dependencies

- Python (>= 2.6)

- scipy
- scikit-learn
- pysam (uses 0.8.3)
- pandas
- numpy (required for scikit-learn and pandas anyway)
- biopython (uses 1.65) 

#Installation

Once the dependencies are installed the scripts should run as they are.

#Scripts/Modules

bedNbamClassifier.py is the main script, it is designed to do supervised classfication of regions of the genome using 
kmers of reads from that region.  It takes a bam or multiple bam files and a list of bed files (at least two) that are 
regions that constitute each class.

bedNbamConsensusClassifier.py is designed to do classification or regions of the genome using kmers from a consensus sequence (NOT recommended)
it will also test a machine against kmers from regions...so is flat, eliminating hte possibility that it is read coverage giving the result.


#Other Notes

Tested on Mac OS X (Yosemite) and Linux (Ubuntu 10.04, 12.04 and 14.04)