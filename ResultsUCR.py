import os
import utils.UCRDatabase as ucr
import numpy as np


errors = []

# set global parameters
squared, symmetric = True, True

print " ", "\t", " "*30, "  Delta +/- Sigma   Delta/Sigma   Ecdtw +/- Sigma       Egem +/- Sigma       Eeuc +/- Sigma       Edtw +/- Sigma"

for datasetN in range(50):
    
    # get files that match dataset and global parameters
    files = list(os.walk("./results/finalresults"))[0][2]
    files = filter(lambda x: ("dn_%s-" % datasetN) in x, files)
    files = filter(lambda x: ("sq_%s" % squared) in x, files)
    files = filter(lambda x: ("sy_%s" % symmetric) in x, files)

    if len(files) == 0:
        continue

    # empty list for the gains in training phase and testing phase
    gainresult, gemresult, cdtwresult, eucresult, dtwresult = [], [], [], [], []

    for filename in files:
        with open("./results/finalresults/%s" % filename, "r") as f:

            bestlp, bestfulldtw, bestconstdtw, bestgem = None, None, None, None

            for line in f:
                if "BESTLP=" in line:
                    bestlp=eval(line.split("=")[1])
                if "BESTFULLDTW=" in line:
                    bestfulldtw=eval(line.split("=")[1])
                if "BESTCONSDTW=" in line:
                    bestconsdtw=eval(line.split("=")[1])
                if "BESTGEM=" in line:
                    bestgem=eval(line.split("=")[1])
            
            gainresult.append((bestconsdtw[2]-bestgem[2]))
            cdtwresult.append(bestconsdtw[2])
            gemresult.append(bestgem[2])
            eucresult.append(bestlp[2])
            dtwresult.append(bestfulldtw[2])
            
    if np.mean(gainresult) < 0:
        print datasetN, "\t", ucr.datasetName(datasetN), "%1.4f +/- %1.4f" % (np.mean(gainresult), np.std(gainresult)), \
              "   %1.4f" % (np.mean(gainresult)/np.std(gainresult)), 
    else:
        print datasetN, "\t", ucr.datasetName(datasetN), " %1.4f +/- %1.4f" % (np.mean(gainresult),np.std(gainresult)), \
              "    %1.4f" % (np.mean(gainresult)/np.std(gainresult)), 
          
    print "   %1.4f +/- %1.4f" % (np.mean(cdtwresult), np.std(cdtwresult)),
    print "   %1.4f +/- %1.4f" % (np.mean(gemresult), np.std(gemresult)),
    print "   %1.4f +/- %1.4f" % (np.mean(eucresult), np.std(eucresult)),
    print "   %1.4f +/- %1.4f" % (np.mean(dtwresult), np.std(dtwresult))
