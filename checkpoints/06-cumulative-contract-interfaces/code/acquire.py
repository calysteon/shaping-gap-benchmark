"""Prepare a fresh local workspace and run captured acquisition sources on CPU."""
from pathlib import Path
import argparse,shutil,subprocess,sys,os
CP=Path(__file__).resolve().parent.parent;PREVIOUS=CP.parent/'05-component-learning-cycle'
STUDIES=['cumulative-recruitment-study','contract-interface-study','application-transfer-study']

def prepare(workspace):
 workspace.mkdir(parents=True,exist_ok=True);(workspace/'results').mkdir(exist_ok=True)
 for name in STUDIES:
  dest=workspace/name
  if not dest.exists():shutil.copytree(CP/'code/source'/name,dest)
 p=workspace/'results/p63-component-acquisition-reuse-001'
 if not p.exists():p.symlink_to(PREVIOUS/'data/experiment',target_is_directory=True)
 dest=workspace/'component-cycle-study';dest.mkdir(exist_ok=True)
 if not (dest/'protocol.json').exists():shutil.copy2(PREVIOUS/'protocols/protocol.json',dest/'protocol.json')
 for name,source in [(STUDIES[0]+'/pools',CP/'data/pools'),(STUDIES[2]+'/data',CP/'data/application')]:
  p=workspace/name
  if not p.exists():p.symlink_to(source,target_is_directory=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('study',choices=['recruitment','contract','application','prepare']);parser.add_argument('--workspace',type=Path,default=CP/'runs/workspace');args=parser.parse_args();workspace=args.workspace.resolve();prepare(workspace)
 if args.study=='prepare':print(workspace);return
 name=STUDIES[['recruitment','contract','application'].index(args.study)]
 if args.study=='application' and not (workspace/'results/p65-program-contract-interfaces-001/summary.json').exists():raise SystemExit('Run contract acquisition in this workspace before application transfer.')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');subprocess.run([sys.executable,str(workspace/name/'run.py')],env=env,check=True)
if __name__=='__main__':main()
