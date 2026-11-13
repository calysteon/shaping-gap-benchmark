"""Create a separate native replication workspace; preserve published records."""
from pathlib import Path
import argparse,json,shutil

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,default=ROOT/'models');p.add_argument('--checkpoint-06',type=Path);p.add_argument('--checkpoint-17',type=Path);a=p.parse_args()
    workspace=ROOT/'runs/native-workspace';workspace.mkdir(parents=True,exist_ok=True)
    for name in ['preserved-return-study','observer-transport-study']:
        shutil.copytree(ROOT/'code/studies'/name,workspace/name,dirs_exist_ok=True)
        for f in (workspace/name).glob('*.py'):
            text=f.read_text().replace("LAB.parent.parent/'outputs/neural-stimulus/preserved-augmentation'","LAB/'replay-output'");f.write_text(text)
    parents=['p55-neural-stimulus-capacity-002','p56-nonlinear-stimulus-development-001']
    for parent in parents:shutil.copytree(ROOT/'data/results'/parent,workspace/'results'/parent,dirs_exist_ok=True)
    specs=[('0.6B','Qwen/Qwen3-0.6B-MLX-4bit','173234aa840d113125e9f2271100ddbaf16c9620',a.checkpoint_06),('1.7B','Qwen/Qwen3-1.7B-MLX-4bit','21457c6f51ed54a7c16e988c0844db973815c137',a.checkpoint_17)]
    for label,repo,revision,local in specs:
        if local:
            path=local.resolve()
            if not (path/'config.json').is_file() or not any(path.glob('*.safetensors')):
                raise FileNotFoundError(f'Missing model config or weights: {path}')
        else:
            from huggingface_hub import snapshot_download
            path=a.models.resolve()/repo.split('/')[-1];snapshot_download(repo_id=repo,revision=revision,local_dir=str(path),allow_patterns=['*.json','*.safetensors'])
        for parent in parents:
            receipt=workspace/'results'/parent/'models'/label/'checkpoint.json';r=json.loads(receipt.read_text());r['local_path']=str(path);receipt.write_text(json.dumps(r,indent=2)+'\n')
    print(workspace)

if __name__=='__main__':main()
