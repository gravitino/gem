import os
import utils.UCRDatabase as ucr
import numpy as np
import pylab as pl

errors = []

# set global parameters
squared, symmetric = True, True

pl.figure(1, figsize=(8, 7))
pl.subplot("111", aspect=1.0)
pl.figure(2, figsize=(8, 7))
pl.subplot("111", aspect=1.0)
pl.rcParams.update({'font.size': 22})

for datasetN in range(50):
    
    # get files that match dataset and global parameters
    files = list(os.walk("./results/finalresults"))[0][2]
    files = filter(lambda x: ("dn_%s-" % datasetN) in x, files)
    files = filter(lambda x: ("sq_%s" % squared) in x, files)
    files = filter(lambda x: ("sy_%s" % symmetric) in x, files)

    if len(files) == 0:
        continue

    # empty list for the gains in training phase and testing phase
    gaintrain, gainresult, cdtwResult, gemResult = [], [], [], []

    for filename in files:
        with open("./results/finalresults/%s" % filename, "r") as f:

            bestlearnconsdtw, bestlearngem = None, None
            bestconstdtw, bestgem = None, None

            for line in f:
                if "BESTLEARNCONSDTW=" in line:
                    bestlearnconsdtw=eval(line.split("=")[1])[0]
                if "BESTLEARNGEM=" in line:
                    bestlearngem=eval(line.split("=")[1])[0]
                if "BESTCONSDTW=" in line:
                    bestconsdtw=eval(line.split("=")[1])
                if "BESTGEM=" in line:
                    bestgem=eval(line.split("=")[1])
            
            gaintrain.append(bestlearnconsdtw[2]-bestlearngem[2])
            gainresult.append(bestconsdtw[2]-bestgem[2])
            cdtwResult.append(bestconsdtw[2])
            gemResult.append(bestgem[2])
    
    pl.figure(1)
    pl.subplot("111", aspect=1.0)
    pl.errorbar([np.mean(gaintrain)], [np.mean(gainresult)], 
                [np.std(gaintrain)], [np.std(gainresult)], c="grey")
    pl.plot([np.mean(gaintrain)], [np.mean(gainresult)], "o", c="blue") 
    
    pl.figure(2)
    pl.subplot("111", aspect=1.0)
    pl.errorbar([np.mean(cdtwResult)], [np.mean(gemResult)], 
                [np.std(cdtwResult)], [np.std(gemResult)], c="grey")
    pl.plot([np.mean(cdtwResult)], [np.mean(gemResult)], "o", c="blue") 


pl.figure(1)
pl.subplot("111", aspect=1.0)
#pl.title("Texas Sharpshooter Plot")
pl.plot([-0.25, 0.25], [0, 0], c="black")
pl.plot([0, 0], [-0.25, 0.25], c="black")
pl.xlabel('relative error difference on training set')
pl.ylabel('relative error difference on test set')
pl.axis((-0.25, 0.25,-0.25, 0.25))
pl.text(+0.2, +0.2, r'TP', fontsize=20)
pl.text(-0.23, -0.23, r'TN', fontsize=20)
pl.text(+0.2, -0.23, r'FP', fontsize=20)
pl.text(-0.23, +0.2, r'FN', fontsize=20)
pl.annotate('Lighting7', xy=(-0.12, -0.1), xytext=(-0.244, -0.04), arrowprops=dict(facecolor='lightgrey', shrink=0.05), color="grey")
pl.annotate('Fish', xy=(0.095, 0.115), xytext=(0.01, 0.16), arrowprops=dict(facecolor='lightgrey', shrink=0.05), color="grey")


pl.figure(2)
pl.subplot("111", aspect=1.0)
#pl.title("Relative Error cDTW vs. GEM")
pl.plot([-0.05, 0.65], [-0.05, 0.65], c="black")
pl.xlabel('relative error for cDTW')
pl.ylabel('relative error for GEM')
pl.axis((0.0, 0.65, 0.0, 0.65))
pl.text(0.19, 0.02, r'In this region GEM is better.', fontsize=20)
pl.text(0.02, 0.65-0.04, r'In this region cDTW is better.', fontsize=20)
pl.annotate('Lighting7', xy=(0.23, 0.38), xytext=(0.04, 0.45), arrowprops=dict(facecolor='lightgrey', shrink=0.05), color="grey")
pl.annotate('Fish', xy=(0.23, 0.09), xytext=(0.32, 0.09), arrowprops=dict(facecolor='lightgrey', shrink=0.05), color="grey")

pl.show()


