"""Prepare a separate complete acquisition workspace from preserved historical data."""
from pathlib import Path
import argparse,json,shutil
ROOT=Path(__file__).resolve().parents[1];PREVIOUS=ROOT.parent/'03-preserved-neural-augmentation'
def main():
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,default=ROOT/'models');p.add_argument('--checkpoint-06',type=Path);p.add_argument('--checkpoint-17',type=Path);a=p.parse_args();workspace=ROOT/'runs/native-workspace'
 if (workspace/'results/p61-learned-update-compatibility-001').exists():raise FileExistsError('Use a new workspace before reacquiring P61; existing results are preserved.')
 for study in ['update-policy-study','feedback-update-study']:shutil.copytree(ROOT/'code/studies'/study,workspace/study,dirs_exist_ok=True)
 for parent in ['p55-neural-stimulus-capacity-002','p59-preserved-return-augmentation-001']:
  path=PREVIOUS/'data/results'/parent
  if not path.exists():raise FileNotFoundError(f'Preserved Checkpoint03 data missing: {path}')
  shutil.copytree(path,workspace/'results'/parent,dirs_exist_ok=True)
 specs=[('0.6B','Qwen/Qwen3-0.6B-MLX-4bit','173234aa840d113125e9f2271100ddbaf16c9620',a.checkpoint_06),('1.7B','Qwen/Qwen3-1.7B-MLX-4bit','21457c6f51ed54a7c16e988c0844db973815c137',a.checkpoint_17)]
 for label,repo,revision,provided in specs:
  path=(provided or a.models/repo.split('/')[-1]).resolve()
  if provided:
   if not (path/'config.json').exists() or not any(path.glob('*.safetensors')):raise FileNotFoundError(path)
  else:
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id=repo,revision=revision,local_dir=str(path),allow_patterns=['*.json','*.safetensors'])
  for parent in ['p55-neural-stimulus-capacity-002','p59-preserved-return-augmentation-001']:
   receipt=workspace/'results'/parent/'models'/label/'checkpoint.json';r=json.loads(receipt.read_text());r['local_path']=str(path);receipt.write_text(json.dumps(r,indent=2)+'\n')
 print(workspace)
if __name__=='__main__':main()
