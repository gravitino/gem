import os
import numpy as np

def read(dirname="./data/metal"):
    """ read data set from Metal database"""

    dr_parents, dr_childs = dirname+"/parent/", dirname+"/child/"

    fn_parents, fn_childs = os.walk(dr_parents), os.walk(dr_childs)
    fn_parents, fn_childs = list(fn_parents)[0][2], list(fn_childs)[0][2]

    parentLabels, parent, childLabels, child = [], [], [], []

    paired = zip(sorted(fn_parents), sorted(fn_childs))

    for index, (fn_parent, fn_child) in enumerate(paired):
    
        s_parent, s_child = [], []
    
        with open(dr_parents+fn_parent, "r") as f:
            for line in f:
                s_parent.append(float(line))
            
        with open(dr_childs+fn_child, "r") as f:
            for line in f:
                s_child.append(float(line))
                
        parentLabels.append(index)
        childLabels.append(index)
         
        parent.append(s_parent)
        child.append(s_child)
         
    return (parentLabels, parent), (childLabels, child)

def znormalize(series):
    """z-normalize a time series"""

    return (np.array(series)-np.mean(series))/np.std(series)

def scale(series, length=2**10):
    """scale series to length"""

    return np.interp(np.linspace(0, 1, length),
                     np.linspace(0, 1, len(series)), series)

if __name__ == '__main__': 

    import pylab as pl
    
    (parentLabels, parent), (childLabels, child) = read()
    
    for plabel, clabel, p, c in zip(parentLabels, childLabels, parent, child):
        
        pl.title("(%s, %s)" %(plabel, clabel))
        pl.plot(p)
        pl.plot(c)
        pl.show()
        
