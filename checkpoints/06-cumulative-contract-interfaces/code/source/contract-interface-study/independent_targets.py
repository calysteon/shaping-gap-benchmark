import numpy as np

def targets(drive,start,n,descriptors):
 idx=np.arange(start,start+n);u=np.stack([drive[idx-k] for k in range(16)])
 # Recurrence is independent of the closed-form formulas in the runner.
 polys={0:np.ones_like(u),1:u}
 for d in range(2,5):polys[d]=((2*d-1)*u*polys[d-1]-(d-1)*polys[d-2])/d
 return np.column_stack([np.prod([np.sqrt(2*d+1)*polys[d][lag] for lag,d in desc['parts']],axis=0) for desc in descriptors])
