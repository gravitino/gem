import numpy as np

def obtain_1NN_error(testLabels, testSet, trainLabels, trainSet, dist_func):
    """calculate 1NN error rate for given distance function"""
    
    errors = []
    
    for queryLabel, query in zip(testLabels, testSet):
        dist = [dist_func(query, subject) for subject in trainSet]
        best = np.array(dist).argmin()
        
        if queryLabel != trainLabels[best]:
            errors.append(1)
        else:
            errors.append(0)
    
    error = sum(errors)
    
    return (error, len(testSet), float(error)/len(testSet)), errors
