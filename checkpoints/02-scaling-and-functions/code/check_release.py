"""Check the repository file manifest and research-checkpoint links (stdlib only)."""
from pathlib import Path
import hashlib
import json
import re


def main():
    root=Path(__file__).resolve().parents[3]
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for row in manifest['files']:
        p=root/row['path']
        assert p.is_file(),row['path']
        assert p.stat().st_size==row['bytes'],row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
    index=json.loads((root/'CHECKPOINTS.json').read_text())
    assert len({r['id'] for r in index['checkpoints']})==len(index['checkpoints'])
    assert index['latest'] in {r['id'] for r in index['checkpoints']}
    for row in index['checkpoints']:
        assert (root/row['paper']).is_file() and (root/row['guide']).is_file()
    missing=[];links=0
    for p in [root/'README.rst',root/'CHANGELOG.rst',*root.glob('checkpoints/*/README.rst')]:
        for target in re.findall(r'<([^>\n]+)>',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            links+=1
            if not (p.parent/target.split('#')[0]).exists():missing.append((str(p),target))
    assert not missing,missing
    print(json.dumps(dict(manifest_files_verified=len(manifest['files']),checkpoint_count=len(index['checkpoints']),relative_navigation_links_verified=links)))


if __name__=='__main__':main()
