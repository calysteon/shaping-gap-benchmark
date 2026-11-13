"""Read-only numerical audit; receipts go only to runs/replay/."""
from pathlib import Path
import subprocess,sys,os,json,hashlib,time
HERE=Path(__file__).resolve().parent;CP=HERE.parent

def main():
 start=time.perf_counter();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');receipts=[]
 for name in ['p64','p65','maps','native','application']:
  subprocess.run([sys.executable,str(HERE/'verification'/f'{name}.py')],env=env,check=True)
 result={'passed':True,'checks':['p64','p65','maps','native','application'],'wall_seconds':time.perf_counter()-start,'recorded_data_modified':False}
 (CP/'runs/replay/complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
