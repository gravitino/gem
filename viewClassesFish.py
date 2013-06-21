import pylab as pl
import numpy as np
import collections as col
import utils.UCRDatabase as ucr

if __name__ == '__main__': 

    # configure canvas
    pl.figure(1, figsize=(21, 7))

    # set fish data set
    datasetN = 39
    
    # read the dataset
    (testLabels, testSet), (trainLabels, trainSet)  = ucr.read(datasetN)
    # merge dataset
    labels, items = ucr.merge(testLabels, testSet, trainLabels, trainSet)

    # reverse mapping for classes
    classes = col.defaultdict(list)
    for index, label in enumerate(labels):
        classes[label].append(index)

    # plot classes as family of five time series
    for label in sorted(classes):
        for index in classes[label][:5]:
            
            pl.plot(items[index])

        pl.axis("off")
        pl.tight_layout()
        pl.show()
