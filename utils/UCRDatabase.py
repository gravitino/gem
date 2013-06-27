import os
import numpy as np

def read(set_number, dirname="./data/ucr"):
    """ read data set from UCR database"""

    data_sets=list(sorted(list(os.walk(dirname))[0][1]))
    
    if not 0 <= set_number < len(data_sets):
        raise Exception, "no such data set (too big or too small set_number?)"
    
    print "data set:", data_sets[set_number]
    
    # construct full path to test and train sets
    path = "%s/%s" % (dirname, data_sets[set_number])

    # get test file and train file name    
    empty, test_file, train_file = list(sorted(list(os.walk(path))[0][2]))

    print "test: ", test_file, "\ttrain: ", train_file

    test, train = [], []
    
    with open("%s/%s" % (path, test_file), "r") as f:
        for line in f:
            test.append(map(float, line.split()))

    with open("%s/%s" % (path, train_file), "r") as g:
        for line in g:
            train.append(map(float, line.split()))

    test, train = map(np.array, [test, train])

    test_labels, train_labels = test[:,0], train[:,0]
    test, train = test[:,1:], train[:,1:]
    
    print "len train: ", len(train), "len test: ", len(test)
    
    return (test_labels, test), (train_labels, train)

def datasetName(set_number, dirname="./data/ucr", tab=30):
    """get the identification string of the dataset"""

    data_sets=list(sorted(list(os.walk(dirname))[0][1]))

    if not 0 <= set_number < len(data_sets):
        raise Exception, "no such data set (too big or too small set_number?)"

    trail= "."*(tab-len(data_sets[set_number]))
    
    return data_sets[set_number]+trail

def merge(testLabels, testSet, trainLabels, trainSet):
    """merge training and test set"""

    labels = np.hstack((testLabels, trainLabels))
    items = np.vstack((testSet, trainSet))

    return labels, items

def znormalize(series):
    """z-normalize a time series"""

    return (np.array(series)-np.mean(series))/np.std(series)
    
if __name__ == '__main__': 

    import pylab as pl
    import sys    

    number = int(sys.argv[1])
    (testLabels, testSet), (trainLabels, trainSet) = read(number)
    
    for label, item in zip(trainLabels, trainSet):
        
        pl.title(label)
        pl.plot(item)
        pl.show()
        
