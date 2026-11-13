from pathlib import Path
import gzip

def open_trace(directory):
    p=Path(directory)/"runs.jsonl"
    return p.open() if p.exists() else gzip.open(p.with_suffix(".jsonl.gz"),"rt")
