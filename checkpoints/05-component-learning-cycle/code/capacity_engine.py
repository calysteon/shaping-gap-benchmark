"""The unchanged143-function temporal-processing target construction."""
import numpy as np

def capacity_targets(drive,start,count):
    idx=np.arange(start,start+count);u=np.array([drive[idx-lag] for lag in range(16)])
    P={1:np.sqrt(3)*u,2:np.sqrt(5)*(3*u*u-1)/2,3:np.sqrt(7)*(5*u**3-3*u)/2,4:3*(35*u**4-30*u*u+3)/8}
    descriptors=[];ys=[]
    def append(parts):
        y=np.ones(count)
        for lag,degree in parts:y*=P[degree][lag]
        descriptors.append(dict(degree=sum(d for _,d in parts),distinct_lags=len(parts),parts=parts));ys.append(y)
    for lag in range(16):append([(lag,1)])
    for lag in range(8):append([(lag,2)])
    from itertools import combinations
    for pair in combinations(range(8),2):append([(j,1) for j in pair])
    for lag in range(6):append([(lag,3)])
    for i in range(6):
        for j in range(6):
            if i!=j:append([(i,2),(j,1)])
    for triple in combinations(range(6),3):append([(j,1) for j in triple])
    for lag in range(4):append([(lag,4)])
    for i in range(4):
        for j in range(4):
            if i!=j:append([(i,3),(j,1)])
    for pair in combinations(range(4),2):append([(j,2) for j in pair])
    for i in range(4):
        for pair in combinations([j for j in range(4) if j!=i],2):append([(i,2),*[(j,1) for j in pair]])
    append([(j,1) for j in range(4)])
    return np.stack(ys,axis=1),descriptors
