import os
import sys
import numpy as np
import pylab as pl

pl.figure(1, figsize=(21, 7))
pl.figure(2, figsize=(21, 7))
pl.rcParams.update({'font.size': 40})

# set global parameters
squared, symmetric = True, True

# get files that match dataset and global parameters
files = list(os.walk("./results/finalresults"))[0][2]
files = filter(lambda x: ("dn_M") in x, files)
files = filter(lambda x: ("sq_%s" % squared) in x, files)
files = filter(lambda x: ("sy_%s" % symmetric) in x, files)

# empty list for the gains in training phase and testing phase
errors, dtw, gem = [], [], []

for filename in files:
    with open("./results/finalresults/%s" % filename, "r") as f:
        for line in f:
            if "BESTCONSDTWONE=" in line:
                bestconsdtwone=eval(line.split("=")[1])
            if "BESTGEMONE=" in line:
                bestgemone=eval(line.split("=")[1])
            if "BESTCONSDTWTWO=" in line:
                bestconsdtwtwo=eval(line.split("=")[1])
            if "BESTGEMTWO=" in line:
                bestgemtwo=eval(line.split("=")[1])
                
    dtw.append(bestconsdtwone[2])
    dtw.append(bestconsdtwtwo[2])
    gem.append(bestgemone[2])
    gem.append(bestgemtwo[2])
    errors.append(bestconsdtwone[2]-bestgemone[2])
    errors.append(bestconsdtwtwo[2]-bestgemtwo[2])

print "E_cDTW +/- sigma = %1.3f +/- %1.3f" % (np.mean(dtw), np.std(dtw))
print "E_GEM  +/- sigma = %1.3f +/- %1.3f" % (np.mean(gem), np.std(gem))
print "Delta  +/- sigma = %1.3f +/- %1.3f     Delta/sigma = %1.3f" % \
      (np.mean(errors), np.std(errors), np.mean(errors)/np.std(errors))

pl.figure(1)
pl.hist(errors)
pl.xlabel('difference of relative errors')
pl.ylabel('number of events')

pl.figure(2)
pl.hist(dtw)
pl.hist(gem)
pl.xlabel('relative error')
pl.ylabel('number of events')
pl.text(0.025, 60, r'GEM', fontsize=40)
pl.text(0.12, 60, r'cDTW', fontsize=40)
pl.axis((0.0, 0.21, 0.0, 140))

pl.tight_layout()
pl.show()
