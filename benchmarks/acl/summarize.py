#!/usr/bin/env python3
import glob, json, statistics
from pathlib import Path

rows=[]
base=Path('results/acl/v2')
status=json.loads((base/'status.json').read_text())
for group in ('upstream','fork','comparators'):
    if group=='fork' and status['fork'] != 'ran':
        continue
    files=sorted(glob.glob(str(base/f'{group}_[123].json')))
    if len(files)!=3:
        raise SystemExit(f'{group}: expected three process results')
    collected={}
    for file in files:
        doc=json.loads(Path(file).read_text())
        for row in doc['results']:
            key=row['runtime']
            if group=='fork' and key.startswith('Rust native'):
                continue
            if group=='fork': key += ' (fork)'
            if row['mismatches']:
                raise SystemExit(f'{file}: correctness mismatch for {key}')
            values=[v/row['checks_per_sample'] for round_ in row['raw_ns'] for v in round_]
            if len(values)!=60: raise SystemExit(f'{file}: incomplete samples')
            entry=collected.setdefault(key, {'samples':[], 'process_medians':[], 'jit_traces':[], 'jit_entries':[]})
            entry['samples'].extend(values)
            entry['process_medians'].append(statistics.median(values))
            entry['jit_traces'].append(row['jit_traces'])
            entry['jit_entries'].append(row['jit_entries'])
    for name,entry in collected.items():
        a=sorted(entry.pop('samples'))
        rows.append({'runtime':name,'median_ns_per_check':round(statistics.median(a),2),
                     'p95_batch_mean_ns_per_check':round(a[round(.95*len(a))-1],2),
                     'median_checks_per_second':round(1e9/statistics.median(a)), **entry})
out={'status':status, 'note':'Batch-average timings include Rust-to-runtime call and conversions; not per-call latency percentiles.', 'results':rows}
(base/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
