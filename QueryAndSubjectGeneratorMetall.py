import sys
import numpy as np
import utils.MetalDatabase as mdb

try:
    import pylab as pl
except:
    print "suppressing plots"


def cat(listOfSeries):
    """concatenate a bunch of time series"""

    return np.hstack(tuple(listOfSeries))

def extract(subject, length):
    """extract a subsequence from subject"""
    
    # get left position of extraction window
    left = np.random.uniform(0, len(subject)-length-1)
    
    return subject[left:left+length]

if __name__ == '__main__': 

    # read data from database
    (parentLabels, parent), (childLabels, child) = mdb.read()
    print "done reading metal data"
    
    # read length
    try:
        L, N  = int(sys.argv[1]), int(sys.argv[2])
    except:
        raise Exception \
              ("python2 QueryAndSubjectGeneratorMetall.py L(int) N(int)")
    
    # for reproducibility
    np.random.seed(L)
    
    # concatenate database
    subject, queries = cat(parent), cat(child)
    
    # write queries to files
    for i in range(N):    
        with open("./data/metal/queries/query_%s_%s" % (L, i), "w") as f:
            query =  extract(queries, L)
            for value in query:
                f.write("%s \n" % value)
            try:
                pl.figure(1)
                pl.plot(query)
            except:
                pass
    
    # write subject
    with open("./data/metal/queries/subject" , "w") as f:
            for value in subject:
                f.write("%s \n" % value)
    try:
        pl.figure(2)
        pl.plot(subject)
        pl.show()
    except:
        pass
