import sys
import numpy as np
import utils.MetalDatabase as mdb
import utils.distances as ds
import utils.learnParameters as lp
import utils.kNNClassifier as cl


if __name__ == '__main__': 

    # global parameters for distance measures (Manhatten/Euclidean, sym. gem)
    squared, symmetric = True, True

    # read split number
    try:
        splitN, splitL = int(sys.argv[1]), int(sys.argv[2])
    except:
        raise Exception \
              ("python2 MetalError.py splitnumber(int) splitlearn(int)")

    # read data from database
    (parentLabels, parent), (childLabels, child) = mdb.read()
    print "done reading metal data"

    # downsample time series to 1024 points to reduce computational complexity
    parent = map(lambda series: mdb.scale(series, length=2**10), parent)
    child = map(lambda series: mdb.scale(series, length=2**10), child)
    print "done with the scaling"

    # znormalize data for all distance measures but gem
    zparent = map(mdb.znormalize, parent)
    zchild = map(mdb.znormalize, child)

    # open file for the logging of results
    f = open("./results/dn_M-sn_%s-lp_%s-sq_%s-sy_%s" % 
            (splitN, splitL, squared, symmetric), "w")
    
    # if splitN == 0 use canoncical split else use random split
    indices = range(len(parent))
    
    if splitN > 0:

        # set seed to splitN for reproducibility
        np.random.seed(splitN)
        np.random.shuffle(indices)

    # partition the data (learn parameters on m pairs for earch halve)
    one = indices[splitL:len(parent)/2]
    two = indices[len(parent)/2+splitL:len(parent)]
    lpone = indices[:splitL]
    lptwo = indices[len(parent)/2:len(parent)/2+splitL] 

    # cast to numpy arrays for convenience
    parentLabels, parent, zparent, childLabels, child, zchild = \
    map(np.array, (parentLabels, parent, zparent, childLabels, child, zchild))
    
    # split the 1st half into parameter learning, test and training dataset
    
    # test (parent_one) and training (child_one)
    parent_one, parentLabels_one = parent[one], parentLabels[one]
    child_one, childLabels_one = child[one], childLabels[one]
    
    # test (parent_lpone) and training (child_lpone) for parameter learning
    parent_lpone, parentLabels_lpone = parent[lpone], parentLabels[lpone]
    child_lpone, childLabels_lpone = child[lpone], childLabels[lpone]
    
    # znormalized variants for dtw
    zparent_one, zparent_lpone = zparent[one], zparent[lpone]
    zchild_one, zchild_lpone = zchild[one], zchild[lpone] 
    
    # split the 2nd half into parameter learning, test and training dataset
    
    # test (parent_two) and training (child_two)
    parent_two, parentLabels_two = parent[two], parentLabels[two]
    child_two, childLabels_two = child[two], childLabels[two]
    
    # test (parent_lptwo) and training (child_ltwo) for parameter learning
    parent_lptwo, parentLabels_lptwo = parent[lptwo], parentLabels[lptwo]
    child_lptwo, childLabels_lptwo = child[lptwo], childLabels[lptwo]
    
    # znormalized variants for dtw
    zparent_two, zparent_lptwo = zparent[two], zparent[lptwo]
    zchild_two, zchild_lptwo = zchild[two], zchild[lptwo] 
    
    # write the split of test and training data to logfile
    f.write("# indices of halve split with split for parameters)\n")
    f.write("INDEXONE=%s\n" % str(one))
    f.write("INDEXLPONE=%s\n" % str(lpone))
    f.write("INDEXTWO=%s\n" % str(two))
    f.write("INDEXLPTWO=%s\n" % str(lptwo))
    f.write("\n")
   
    print "######################### Learn Parameters ########################"

    # learn parameters for gem and constrained dtw with loocv
    best_dtw_one, l_dtw_one = \
    lp.learn_metal_cdtw(parentLabels_lpone, zparent_lpone, 
                        childLabels_lpone, zchild_lpone, symmetric, squared)
    best_dtw_two, l_dtw_two = \
    lp.learn_metal_cdtw(parentLabels_lptwo, zparent_lptwo, 
                        childLabels_lptwo, zchild_lptwo, symmetric, squared)
                         
    best_gem_one, l_gem_one = \
    lp.learn_metal_gem(parentLabels_lpone, parent_lpone, 
                       childLabels_lpone, child_lpone, symmetric, squared)
    best_gem_two, l_gem_two = \
    lp.learn_metal_gem(parentLabels_lptwo, parent_lptwo, 
                       childLabels_lptwo, child_lptwo, symmetric, squared)
    
    print "learned parameter for dtw\n", best_dtw_one, "\n", best_dtw_two
    print "learned parameter for gem\n", best_gem_one, "\n", best_gem_two
    
    # write learned parameters to logfile
    f.write("# learned parameters for cdtw and gem\n")
    f.write("# dtw ((error, size, error/size), (window, sqr))\n")
    f.write("# gem ((error, size, error/size), (St0, St1, E, sym, sqr))\n")
    f.write("BESTLEARNCONSDTWONE=%s\n" % str(best_dtw_one))
    f.write("LISTLEARNCONSDTWONE=%s\n\n" % str(l_dtw_one))  
    f.write("BESTLEARNGEMONE=%s\n" % str(best_gem_one))
    f.write("LISTLEARNGEMONE=%s\n\n" % str(l_gem_one))
    f.write("BESTLEARNCONSDTWTWO=%s\n" % str(best_dtw_two))
    f.write("LISTLEARNCONSDTWTWO=%s\n\n" % str(l_dtw_two))  
    f.write("BESTLEARNGEMTWO=%s\n" % str(best_gem_two))
    f.write("LISTLEARNGEMTWO=%s\n\n" % str(l_gem_two))
    f.write("\n")
   

    print "######################### Calculate Errors ########################"
    
    # write error rates to logging file
    f.write("# error rates for different distance measures\n")
    f.write("# (error, size, error/size) and binary mask\n")
    
    # obtain error for lp-norm
    dist = ds.euc if squared else ds.man
    e, l = cl.obtain_1NN_error(parentLabels_one, zparent_one, 
                               childLabels_one, zchild_one, dist)
    
    print "BESTLPONE=%s\n" % str(e)
    f.write("BESTLPONE=%s\n" % str(e))
    f.write("LISTLPONE=%s\n\n" % str(l))
    
    dist = ds.euc if squared else ds.man
    e, l = cl.obtain_1NN_error(parentLabels_two, zparent_two, 
                               childLabels_two, zchild_two, dist)
    
    print "BESTLPTWO=%s\n" % str(e)
    f.write("BESTLPTWO=%s\n" % str(e))
    f.write("LISTLPTWO=%s\n\n" % str(l))
    
    
    # obtain error for unconstrained dtw
    dist = lambda query, subject: ds.dtw(query, subject, squared)
    e, l = cl.obtain_1NN_error(parentLabels_one, zparent_one, 
                               childLabels_one, zchild_one, dist)
    
    print "BESTFULLDTWONE=%s\n" % str(e)
    f.write("BESTFULLDTWONE=%s\n" % str(e))
    f.write("LISTFULLDTWONE=%s\n\n" % str(l))
    
    dist = lambda query, subject: ds.dtw(query, subject, squared)
    e, l = cl.obtain_1NN_error(parentLabels_two, zparent_two, 
                               childLabels_two, zchild_two, dist)
    
    print "BESTFULLDTWTWO=%s\n" % str(e)
    f.write("BESTFULLDTWTWO=%s\n" % str(e))
    f.write("LISTFULLDTWTWO=%s\n\n" % str(l))
    
    # obtain error for constrained dtw
    window = int(np.round(best_dtw_one[1][0]*len(parent[0])))
    dist = lambda query, subject: ds.cdtw(query, subject, window, squared)
    e, l = cl.obtain_1NN_error(parentLabels_one, zparent_one, 
                               childLabels_one, zchild_one, dist)
    
    print "BESTCONSDTWONE=%s\n" % str(e)
    f.write("BESTCONSDTWONE=%s\n" % str(e))
    f.write("LISTCONSDTWONE=%s\n\n" % str(l))
    
    window = int(np.round(best_dtw_two[1][0]*len(parent[0])))
    dist = lambda query, subject: ds.cdtw(query, subject, window, squared)
    e, l = cl.obtain_1NN_error(parentLabels_two, zparent_two, 
                               childLabels_two, zchild_two, dist)
    
    print "BESTCONSDTWTWO=%s\n" % str(e)
    f.write("BESTCONSDTWTWO=%s\n" % str(e))
    f.write("LISTCONSDTWTWO=%s\n\n" % str(l))
    
    
    # obtain error for gem
    St0, St1, E = best_gem_one[1][:3]
    dist = lambda query, subject: \
                        ds.gem(query, subject, St0, St1, E, symmetric, squared)
    e, l = cl.obtain_1NN_error(parentLabels_one, parent_one, 
                               childLabels_one, child_one, dist)
    
    print "BESTGEMONE=%s\n" % str(e)
    f.write("BESTGEMONE=%s\n" % str(e))
    f.write("LISTGEMONE=%s\n\n" % str(l))
    
    St0, St1, E = best_gem_two[1][:3]
    dist = lambda query, subject: \
                        ds.gem(query, subject, St0, St1, E, symmetric, squared)
    e, l = cl.obtain_1NN_error(parentLabels_two,parent_two, 
                               childLabels_two, child_two, dist)
    
    print "BESTGEMTWO=%s\n" % str(e)
    f.write("BESTGEMTWO=%s\n" % str(e))
    f.write("LISTGEMTWO=%s\n\n" % str(l))
    
    # close the log file
    f.close()


