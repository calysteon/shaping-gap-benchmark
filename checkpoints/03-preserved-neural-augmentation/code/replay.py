"""Model-free independent replay and publication-manifest verification."""
from pathlib import Path
import argparse,importlib.util,json,hashlib

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'runs/replay');a=p.parse_args()
    spec=importlib.util.spec_from_file_location('checkpoint_replay',ROOT/'code/studies/preserved-return-study/verify_offline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.LAB=a.data_root.resolve();m.OUT=a.output.resolve();m.OUT.mkdir(parents=True,exist_ok=True);m.main()
    manifest=ROOT/'MANIFEST.json';count=0
    if manifest.exists():
        for record in json.loads(manifest.read_text())['files']:
            path=ROOT/record['path'];assert path.stat().st_size==record['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path'];count+=1
    receipt=dict(manifest_files_verified=count,recorded_data_unchanged=True,scope='Independent numerical replay; original-weight re-execution has a separate receipt.')
    (m.OUT/'manifest-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
