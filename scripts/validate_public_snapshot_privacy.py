#!/usr/bin/env python3
"""Validate public research snapshot digests and exclude private meeting source locators."""
import hashlib, json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'PUBLICATION_MANIFEST.json').read_text())
files=manifest['files']
assert len(files)>=1000
seen=set()
for x in files:
    p=x['path']
    assert p not in seen,p
    seen.add(p)
    f=R/p
    assert f.is_file(),p
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest==x['published_sha256'],p
for p in [
    'configs/experiment/current-campaign.json',
    'configs/experiment/designs/post-advisor-unanswered-202609282338.json',
    'configs/experiment/advisor-questions/q06-work-package.v1.json',
    'configs/experiment/advisor-questions/q08-work-package-202610012252.json'
]:
    assert p in seen,p
    body=(R/p).read_text(encoding='utf-8')
    assert 'liaochang' not in body.casefold(),p
    assert 'docs/evidence/advisor/liaochang' not in body,p
    assert next(x for x in files if x['path']==p)['transformed'] is True,p
print(f'PUBLICATION_PRIVACY_AND_DIGESTS_PASS: {len(files)} files verified')
