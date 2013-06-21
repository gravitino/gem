import numpy as np
import pylab as pl
import scipy.cluster.hierarchy as h
import utils.distances as ds
import utils.MetalDatabase as mdb

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

 # taken from dn_M-sn_0-lp_100-sq_True-sy_True LISTCONSDTWONE
mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

 # taken from dn_M-sn_0-lp_100-sq_True-sy_True LISTGEMONE
Mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# get misclassified time series
for index in zip(*filter(lambda (i, v): v[0] == 1 and v[1] == 0, 
                 enumerate(zip(mask, Mask))))[0][2:]:

    pl.figure(1, figsize=(21, 14))
    ax = pl.subplot("211")

    # get query
    query = zparent[100+index]

    # taken from dn_M-sn_0-lp_100-sq_True-sy_True BESTLEARNCONSDTWONE
    window = int(np.round(0.01*len(parent[0])))
    dist_func= lambda query, subject: ds.cdtw(query, subject, window, True)

    # search nearest neighbor for cdtw
    dist = [dist_func(query, subject) for subject in zchild[100:500]]
    best = np.array(dist).argmin()

    # entries of distance matrix
    AB = ds.cdtw(query, zchild[100+index], window, True)
    AC = np.array(dist).min()
    BC = ds.cdtw(zchild[100+best], zchild[100+index], window, True)
    
    print index, best, AC, AB
    
    # distance matrix
    M = np.array([[0, AB, AC], [AB, 0, BC], [AC, BC, 0]])
    
    # label function
    L = lambda x: {0: "P", 1: "C", 2: "L"}[int(x)]
    
    # render dendrogram
    D = h.dendrogram(h.complete(M), orientation="left", leaf_label_func=L, 
                     link_color_func=lambda k: "b", leaf_font_size=40)
    
    # adjust clipping
    pl.axis((-2**10-200, np.max(D["dcoord"])*1.2, 0, 30))
    
    # colors for time signals
    C = {"P": "b", "C": "b", "L": "r"}
    
    # list of signals
    signals = {"P": query, "C": zchild[100+index], "L": zchild[100+best]}

    # plot signals
    for offset, label in enumerate(D["ivl"]):
        pl.plot(range(-2**10-100, -100), 
                signals[label]+offset*10+5, c=C[label])
   
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_xticks([])
    #pl.axis("off")
    pl.tight_layout()
        

    # now the same for gem
    #pl.figure(2, figsize=(21, 7))
    ax = pl.subplot("212")

    # get query
    query = parent[100+index]

    # taken from dn_M-sn_0-lp_100-sq_True-sy_True BESTLEARNGEMONE
    St0, St1, E = 2, 2, 0.0625
    
    # entries of distance matrix
    AB = ds.gem(query, child[100+index], St0, St1, E, True, True)
    AC = ds.gem(query, child[100+best], St0, St1, E, True, True)
    BC = ds.gem(child[100+best], child[100+index], St0, St1, E, True, True)
    
    print index, best, AC, AB
    
    # distance matrix (rescale to cdtw errors)
    M = np.array([[0, AB, AC], [AB, 0, BC], [AC, BC, 0]])*np.max(M)/np.max([AB, AC, BC])
        
    # label function
    L = lambda x: {0: "P", 1: "C", 2: "L"}[int(x)]
    
    # render dendrogram
    D = h.dendrogram(h.complete(M), orientation="left", leaf_label_func=L, 
                     link_color_func=lambda k: "b", leaf_font_size=40)
    
    # adjust clipping
    pl.axis((-2**10-200, np.max(D["dcoord"])*1.2, 0, 30))
    
    # colors for time signals
    C = {"P": "b", "C": "b", "L": "r"}
    
    # list of signals
    signals = {"P": zparent[100+index], "C": zchild[100+index], "L": zchild[100+best]}

    # plot signals
    for offset, label in enumerate(D["ivl"]):
        pl.plot(range(-2**10-100, -100), 
                signals[label]+offset*10+5, c=C[label])
    

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_xticks([])
    #pl.axis("off")
    pl.tight_layout()
    
    pl.savefig("results/dendrograms/dendrogram_cdtw_vs_gem_%s.png"%(index+100))
    pl.close()
    
    # pl.show()
