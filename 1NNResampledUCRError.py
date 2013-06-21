import sys
import numpy as np
import utils.stratify as st
import utils.distances as ds
import utils.UCRDatabase as ucr
import utils.learnParameters as lp
import utils.kNNClassifier as cl


if __name__ == '__main__': 

    # global parameters for distance measures (Manhatten/Euclidean, sym. gem)
    squared, symmetric = True, True

    # read dataset number and split number
    try:
        datasetN, splitN = int(sys.argv[1]), int(sys.argv[2])
    except:
        raise Exception \
             ("python2 1NNResampledTest.py datasetnumber(int) splitnumber(int)")
    
    # read the dataset
    (testLabels, testSet), (trainLabels, trainSet)  = ucr.read(datasetN)
    # merge dataset
    labels, items = ucr.merge(testLabels, testSet, trainLabels, trainSet)
    
    # open file for the logging of results
    f = open("./results/dn_%s-sn_%s-sq_%s-sy_%s" % 
            (datasetN, splitN, squared, symmetric), "w")
    
    # if splitN == 0 use UCR-split else use random split
    if splitN > 0:

        # determine split ratio from UCR canonical split and resample
        rho = float(len(trainSet))/(len(trainSet)+len(testSet))
        sss = st.StratifiedShuffleSplit(labels, 500, 
                                        test_size=rho, random_state=0)
        test_index, train_index = list(sss)[splitN-1]
        
        # create training and test set
        trainLabels, testLabels = labels[train_index], labels[test_index]
        trainSet, testSet = items[train_index], items[test_index]
        
    else:
        # indices for UCR split
        test_index = range(len(items))[:len(testSet)]
        train_index = range(len(items))[len(testSet):]

    # write the split of test and training data to logfile
    f.write("# shuffled stratified split (indices referring to UCR database)\n")
    f.write("TESTINDEX=%s\n" % str(test_index))
    f.write("TRAININDEX=%s\n" % str(train_index))
    f.write("\n")

    print "######################### Learn Parameters ########################"

    # learn parameters for gem and constrained dtw with loocv
    best_dtw, l_dtw = lp.learn_cdtw(trainLabels, trainSet, squared)
    best_gem, l_gem = lp.learn_gem(trainLabels, trainSet, symmetric, squared)
    
    print "learned parameter for dtw", best_dtw
    print "learned parameter for gem", best_gem
    
    # write learned parameters to logfile
    f.write("# learned parameters for cdtw and gem\n")
    f.write("# dtw ((error, size, error/size), (window, sqr))\n")
    f.write("# gem ((error, size, error/size), (St0, St1, E, sym, sqr))\n")
    f.write("BESTLEARNCONSDTW=%s\n" % str(best_dtw))
    f.write("LISTLEARNCONSDTW=%s\n\n" % str(l_dtw))
    f.write("BESTLEARNGEM=%s\n" % str(best_gem))
    f.write("LISTLEARNGEM=%s\n\n" % str(l_gem))
    f.write("\n")
    
    print "######################### Calculate Errors ########################"
    
    # write error rates to logging file
    f.write("# error rates for different distance measures\n")
    f.write("# (error, size, error/size) and binary mask\n")
    
    # obtain error for lp-norm
    dist = ds.euc if squared else ds.man
    E, L = cl.obtain_1NN_error(testLabels, testSet, trainLabels, trainSet, dist)
    
    print "BESTLP=%s\n" % str(E)
    f.write("BESTLP=%s\n" % str(E))
    f.write("LISTLP=%s\n\n" % str(L))
    
    # obtain error for unconstrained dtw
    dist = lambda query, subject: ds.dtw(query, subject, squared)
    E, L = cl.obtain_1NN_error(testLabels, testSet, trainLabels, trainSet, dist)
    
    print "BESTFULLDTW=%s\n" % str(E)
    f.write("BESTFULLDTW=%s\n" % str(E))
    f.write("LISTFULLDTW=%s\n\n" % str(L))
    
    # obtain error for constrained dtw
    window = int(np.round(best_dtw[1][0]*len(trainSet[0])))
    dist = lambda query, subject: ds.cdtw(query, subject, window, squared)
    E, L = cl.obtain_1NN_error(testLabels, testSet, trainLabels, trainSet, dist)
    
    print "BESTCONSDTW=%s\n" % str(E)
    f.write("BESTCONSDTW=%s\n" % str(E))
    f.write("LISTCONSDTW=%s\n\n" % str(L))
    
    # obtain error for gem
    St0, St1, E = best_gem[1][0], best_gem[1][1], best_gem[1][2]
    dist = lambda query, subject: \
                        ds.gem(query, subject, St0, St1, E, symmetric, squared)
    E, L = cl.obtain_1NN_error(testLabels, testSet, trainLabels, trainSet, dist)
    
    print "BESTGEM=%s\n" % str(E)
    f.write("BESTGEM=%s\n" % str(E))
    f.write("LISTGEM=%s\n\n" % str(L))
    
    # close the log file
    f.close()
    
