import time
import numpy as np
import pylab as pl

import libgem as gem
import libgemcuda as gemc

np.random.seed(0)

def get_signal(length, beta=0.001):
    """coloured noise"""
    K = np.exp(-beta*np.linspace(0, 1, length))
    S = np.random.random(length)
    
    return np.fft.irfft(np.fft.rfft(K)*np.fft.rfft(S))

def znormalize(series):
    """z-normalize series"""

    return (series-np.mean(series))/np.std(series)

# generate query and subject
S = znormalize(get_signal(1001000))
N = gem.TimeSeries(S[:1000])
H = gem.TimeSeries(S[1000:])

print len(N), len(H)

# matching parameters
St0, St1, E = 2, 2, 0.01

R = gem.Result()
t=time.time()
gemc.cuda_match(N, H, R, St0, St1, E, True)
print "time needed:", time.time()-t, "best match:", R[0].penalty, R[0].left, R[0].right

R = gem.Result()
t=time.time()
gem.match(N, H, R, St0, St1, E, True, True)
print "time needed:", time.time()-t, "best match:", R[0].penalty, R[0].left, R[0].right

pl.plot(np.array(N) + 1)
pl.plot(H, c = "lightgrey")

for item in R[:10]:
    L = gem.TimeSeries()
    X = gem.TimeCoords()
    Y = gem.TimeSeries()
    
 
    gem.backtrace(N, H, item, L, X, Y, St0, St1, E, False, True)
    pl.plot(X, Y, color="r")
    pl.plot(X, L, color="b")


pl.show()

