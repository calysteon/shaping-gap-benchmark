"""Verify publication integrity and independently replay the complete experiments."""
from pathlib import Path
import argparse,sys,json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--checkpoint-06',type=Path);p.add_argument('--checkpoint-17',type=Path);p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'runs/replay');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 manifest=ROOT/'MANIFEST.json';count=0
 if manifest.exists():
  for r in json.loads(manifest.read_text())['files']:
   path=ROOT/r['path'];assert path.stat().st_size==r['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'],r['path'];count+=1
 sys.path.insert(0,str(ROOT/'code/studies/update-policy-study'))
 if a.native:
  import native
  original=native.NativeField.__init__;mapping={}
  for label,provided in [('0.6B',a.checkpoint_06),('1.7B',a.checkpoint_17)]:
   path=(provided or ROOT/'models'/('Qwen3-'+label+'-MLX-4bit')).resolve()
   if not (path/'config.json').exists():raise FileNotFoundError(f'Model missing: {path}; provide --checkpoint-06 and --checkpoint-17')
   receipt=json.loads((a.data_root/'results/p61-learned-update-compatibility-001/models'/label/'checkpoint.json').read_text());mapping[receipt['local_path']]=str(path)
  def init(self,path,*args,**kwargs):return original(self,mapping.get(str(path),path),*args,**kwargs)
  native.NativeField.__init__=init
 for i,study in enumerate(['update-policy-study','feedback-update-study']):
  spec=importlib.util.spec_from_file_location('replay_'+str(i),ROOT/'code/studies'/study/'verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.LAB=a.data_root.resolve();m.RUN=m.LAB/'results'/(['p61-learned-update-compatibility-001','p62-feedback-acquired-update-reuse-001'][i]);m.OUT=a.output.resolve()
  if i:m.PARENT=m.LAB/'results/p61-learned-update-compatibility-001'
  saved=sys.argv;sys.argv=[saved[0]]+(['--native'] if a.native else [])
  try:m.main()
  finally:sys.argv=saved
 receipt=dict(publication_manifest_files_verified=count,recorded_data_unchanged=True);(a.output/'manifest-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
