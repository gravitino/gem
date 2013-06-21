import numpy as np
import distances as dist
import utils.kNNClassifier as c

def loocv_cdtw(loocvLabels, loocvSet, relative_window, squared):
    """calculate LOOCV error rate for given window  on a given set"""
    
    error, window = 0, int(np.round(relative_window*len(loocvSet[0])))
    
    # 1NN with Leave One Out Cross Validation
    for index, (queryLabel, query) in enumerate(zip(loocvLabels, loocvSet)):
        search = np.vstack((loocvSet[:index], loocvSet[index+1:]))
        labels = np.hstack((loocvLabels[:index], loocvLabels[index+1:]))

        vals =[dist.cdtw(query, subject, window, squared) for subject in search]
        best = np.array(vals).argmin()
        
        if queryLabel != labels[best]:
            error +=1
    
    return error, len(loocvSet), float(error)/len(loocvSet)

def loocv_gem(loocvLabels, loocvSet, St0, St1, E, symmetric, squared):
    """calculate LOOCV error rate for given parameters on a given set"""

    error = 0

    # 1NN with Leave One Out Cross Validation
    for index, (queryLabel, query) in enumerate(zip(loocvLabels, loocvSet)):
        search = np.vstack((loocvSet[:index], loocvSet[index+1:]))
        labels = np.hstack((loocvLabels[:index], loocvLabels[index+1:]))

        vals =[dist.gem(query, subject, St0, St1, E, symmetric, squared) 
               for subject in search]
        best = np.array(vals).argmin()
        
        if queryLabel != labels[best]:
            error +=1
    
    return error, len(loocvSet), float(error)/len(loocvSet)

def learn_cdtw(trainLabels, trainSet, squared, parameters=None):
    """learn optimal window size for constrained DTW"""
    
    if parameters == None:
        parameters = np.array(range(21))*0.01
    
    loocvLabels, loocvSet = trainLabels, trainSet
    errors = []
    
    for relative_window in parameters:
        result = loocv_cdtw(loocvLabels, loocvSet, relative_window, squared)
        errors.append((result, (relative_window, squared)))
        print errors[-1]

    errs = list(sorted(errors, key=lambda x: x[0][0]))
    
    # pick only the best and take parameter in the "middle"
    best = filter(lambda (x, y): x[0]==errs[0][0][0], errs)
    best.sort(key=lambda (x, y): y[0])
    best = best[len(best)/2]
   
    return best, errs

def learn_gem(trainLabels, trainSet, symmetric, squared, parameters=None):
    """learn optimal parameters for gem"""
    
    if parameters == None:
        ST0 = [1, 2]
        ST1 = [1, 2]
        TAU = [0,0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5]
    else:
        ST0, ST1, TAU = parameters
    
    loocvLabels, loocvSet = trainLabels, trainSet
    errors = []
    
    for St0 in ST0:
        for St1 in ST1:
            for E in TAU :
                result = loocv_gem(loocvLabels, loocvSet, 
                                   St0, St1, E, symmetric, squared)
                errors.append((result, (St0, St1, E, symmetric, squared)))
                print errors[-1]

    errs = list(sorted(errors, key=lambda x: x[0][0]))
    
    # pick only the best and take parameter in the "middle"
    best = filter(lambda (x, y): x[0]==errs[0][0][0], errs)
    best.sort(key=lambda (x, y): y[2])
    best = best[len(best)/2]
   
    return best, errs


def learn_metal_cdtw(parentLabels, parent, childLabels, child, 
                     symmetric, squared, parameters=None):
    """learn optimal parameters for gem on metal data"""
    
    if parameters == None:
        parameters = np.array(range(21))*0.01
        
    errors = []
    

    for relative_window in parameters:
    
        window = int(np.round(len(parent[0])*relative_window))
    
        func = lambda query, subject: dist.cdtw(query, subject, window, squared)
        e, l = c.obtain_1NN_error(parentLabels, parent, 
                                  childLabels, child, func)
        errors.append((e, (relative_window, squared)))
        print errors[-1]
        
    errs = list(sorted(errors, key=lambda x: x[0][0]))
    
    # pick only the best and take parameter in the "middle"
    best = filter(lambda (x, y): x[0]==errs[0][0][0], errs)
    best.sort(key=lambda (x, y): y[0])
    best = best[len(best)/2]
   
    return best, errs


def learn_metal_gem(parentLabels, parent, childLabels, child, 
                    symmetric, squared, parameters=None):
    """learn optimal parameters for gem on metal data"""
   
    if parameters == None:
        ST0 = [1, 2]
        ST1 = [1, 2]
        TAU = [0,0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5]
    else:
        ST0, ST1, TAU = parameters
   
    errors = []
   
    for St0 in ST0:
        for St1 in ST1:
            for E in TAU :
                func = lambda query, subject: \
                       dist.gem(query, subject, St0, St1, E, symmetric, squared)
                e, l = c.obtain_1NN_error(parentLabels, parent, 
                                          childLabels, child, func)

                errors.append((e, (St0, St1, E, symmetric, squared)))
                print errors[-1]
                
    errs = list(sorted(errors, key=lambda x: x[0][0]))
    
    # pick only the best and take parameter in the "middle"
    best = filter(lambda (x, y): x[0]==errs[0][0][0], errs)
    best.sort(key=lambda (x, y): y[2])
    best = best[len(best)/2]
   
    return best, errs


if __name__ == "__main__":

    import UCRDatabase as ucr
    import sys
    
    for number in [int(sys.argv[1])]:
        # read data set from UCR database
        (testLabels, testSet), (trainLabels, trainSet)  = ucr.read(number)
    
        # z-normalize
        testSet  = np.array(map(ucr.znormalize, testSet))
        trainSet = np.array(map(ucr.znormalize, trainSet))
    
        best_dtw = learn_cdtw(trainLabels, trainSet, False)[0]
        best_gem = learn_gem(trainLabels, trainSet, False, False)[0]

        print "dtw params", best_dtw
        print "gem params", best_gem
