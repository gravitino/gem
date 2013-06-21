import sys

# add library paths
sys.path.append("./pydtw")
sys.path.append("./pygem")

sys.path.append("../pydtw") # local testing
sys.path.append("../pygem") # local testing

try:
    import libdtw as ldtw
except:
    raise ImportError("pydtw missing (build it?)")

try:
    import libgem as lgem
except:
    raise ImportError("pygem missing (build it?)")

try:
    import libgemcuda as lgemc # comment this to drop cuda support
except:
    raise ImportError("pycudagem missing (build it?)")


def dtw(query, subject, squared=True):
    """unconstrained Euclidean-flavoured DTW """
    
    return ldtw.dist_dtw(ldtw.TimeSeries(query), ldtw.TimeSeries(subject), squared)
    
def cdtw(query, subject, window, squared=True):
    """constrained Euclidean-flavoured DTW """
    
    return ldtw.dist_cdtw(ldtw.TimeSeries(query), ldtw.TimeSeries(subject), window, squared)
    
def dtwd(query, subject, squared=True):
    """Keogh's Euclidean-flavoured DTW-ED ratio"""
    
    ED = euc(query, subject) if squared else man(query, subject)
    DTW = dtw(query, subject, squared)
    
    return 0 if ED == 0 else DTW/ED
    

def euc(query, subject):
    """Euclidean metric squared """
    
    return ldtw.dist_euclidean(ldtw.TimeSeries(query), ldtw.TimeSeries(subject))

def man(query, subject):
    """Manhattan metric"""
    
    return ldtw.dist_manhatten(ldtw.TimeSeries(query), ldtw.TimeSeries(subject))
    
def gem(query, subject, St0=2, St1=2, E=0.01, symmetric=False, squared=True):
    """GEM local similarity measure"""

    # enable openmp support
    omp = False

    N = lgem.TimeSeries(query)
    H = lgem.TimeSeries(subject)
    R = lgem.Result()

    lgem.match(N, H, R, St0, St1, E, omp, squared)
    dist = R[0].penalty

    if (symmetric):
        R = lgem.Result()
        lgem.match(H, N, R, St0, St1, E, omp, squared)
        dist = min(dist, R[0].penalty)
    
    return dist

def gpugem(query, subject, St0=2, St1=2, E=0.01, symmetric=False, squared=True):
    """GEM local similarity measure"""

    N = lgemc.TimeSeries(query)
    H = lgemc.TimeSeries(subject)
    R = lgemc.Result()

    lgemc.cuda_match(N, H, R, St0, St1, E, squared)
    dist = R[0].penalty

    if (symmetric):
        R = lgemc.Result()
        lgemc.cuda_match(H, N, R, St0, St1, E, squared)
        dist = min(dist, R[0].penalty)
    
    return dist

if __name__ == "__main__":

    import numpy as np
    import pylab as pl
    
    support = np.linspace(0, 4*np.pi, 1024)
    query, subject = np.sin(support), np.cos(support)

    print "Manhattan distance:", man(query, subject)
    print "Manhattan constrained DTW:", cdtw(query, subject, 16, squared=False)
    print "Manhattan unconstrained DTW:", dtw(query, subject, squared=False)
    print "Manhattan Keogh DTWD", dtwd(query, subject, squared=False)
    print "Manhattan gem:", gem(query, subject, symmetric=False, squared=False)
    print "Manhattan sgem:", gem(query, subject, symmetric=True, squared=False)
    
    print "Euclidean distance:", euc(query, subject)
    print "Euclidean constrained DTW:", cdtw(query, subject, 16, squared=True)
    print "Euclidean unconstrained DTW:", dtw(query, subject, squared=True)
    print "Euclidean Keogh DTWD", dtwd(query, subject, squared=True)
    print "Euclidean gem:", gem(query, subject, symmetric=False, squared=True)
    print "Euclidean sgem:", gem(query, subject, symmetric=True, squared=True)

    print "cuda gem:", gpugem(query, subject, symmetric=False, squared=True)
    print "cuda sgem:", gpugem(query, subject, symmetric=True, squared=True)

    pl.plot(query)
    pl.plot(subject)
    
    pl.show()
