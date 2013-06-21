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
files = filter(lambda x: ("dn_39-") in x, files)
files = filter(lambda x: ("sq_%s" % squared) in x, files)
files = filter(lambda x: ("sy_%s" % symmetric) in x, files)

# empty list for the gains in training phase and testing phase
errors, dtw, gem = [], [], []

for filename in files:
    with open("./results/finalresults/%s" % filename, "r") as f:
        for line in f:
            if "BESTCONSDTW=" in line:
                bestconsdtw=eval(line.split("=")[1])
            if "BESTGEM=" in line:
                bestgem=eval(line.split("=")[1])

                
    dtw.append(bestconsdtw[2])
    gem.append(bestgem[2])
    errors.append(bestconsdtw[2]-bestgem[2])

print np.mean(errors), np.std(errors), np.mean(errors)/np.std(errors), len(errors)

pl.figure(1)
pl.hist(errors)
pl.xlabel('difference of relative errors')
pl.ylabel('number of events')

pl.figure(2)
pl.hist(dtw)
pl.hist(gem)
pl.xlabel('relative error')
pl.ylabel('number of events')
pl.text(0.08, 70, r'GEM', fontsize=40)
pl.text(0.19, 70, r'cDTW', fontsize=40)
pl.axis((0.03, 0.31, 0.0, 90))

pl.tight_layout()
pl.show()
