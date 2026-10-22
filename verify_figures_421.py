"""Validate exact Word figure assets and their 421-report analytical basis."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
from generate_manuscript_figures import BASE_DIR, SOURCE, PRACTICES, profiles, load_data

EXPECTED_PRACTICES = {'External validation':25, 'Multisource integration':74, 'Explainability (broad)':128, 'Cross-site / cross-dataset evaluation':17}


def main():
    df=load_data(SOURCE)
    practices={display:int((df[field]=='Reported').sum()) for field,display in PRACTICES}
    if practices!=EXPECTED_PRACTICES:
        raise AssertionError(f'RQ3 totals differ: {practices} vs {EXPECTED_PRACTICES}')
    assert len(profiles(df))==17 and sum(n for _,n in profiles(df))==275
    manifest=json.loads((BASE_DIR/'figures_manuscript_421/manifest.json').read_text(encoding='utf8'))
    assert manifest['included_reports']==421
    for fig in range(2,9):
        assets=manifest['figures'][str(fig)]
        assert 'pdf' in assets and ('png' in assets or 'svg' in assets)
        for fmt,v in assets.items():
            p=BASE_DIR/'figures_manuscript_421'/v['file']
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=v['sha256']:
                raise AssertionError('Frozen figure missing/modified: '+str(p))
    print('Validated: 421 unique reports, 17 profiles / 275 reports, RQ3=',practices)
    print('Frozen Word figures 2-8: all available asset hashes match.')
    print('PRISMA / Figure 1 is deliberately not certified.')

if __name__=='__main__':main()
