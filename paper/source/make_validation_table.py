"""Expose both measured parts of the effect/context interface in the paper."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
source=ROOT/'validation-data.json'
if not source.exists():source=ROOT.parent/'results/p47-keystone-stop-aware-001/summary.json'
data=json.loads(source.read_text())['validation']
rows=[r'\begin{tabular}{lrrrr}',r'\toprule',
      r'\multicolumn{5}{c}{Accumulated effect $E$: RMSE}\\',
      r'Domain & Differential & Endpoint32 & Query-matched & Structured\\',r'\midrule']
for i,v in enumerate(data,1):
    values=[v[k]['effect_rmse'] for k in ['differential','value_same','value_budget','analytic']]
    rows.append(str(i)+' & '+' & '.join(f'{x:.4g}' for x in values)+r' \\')
rows += [r'\midrule',r'\multicolumn{5}{c}{Context partial $KD$: RMSE and native scale}\\',
         r'Domain & Differential & Endpoint32 & Query-matched & Native RMS\\',r'\midrule']
for i,v in enumerate(data,1):
    values=[v[k]['effect_derivative_rmse'] for k in ['differential','value_same','value_budget']]
    values.append(v['differential']['effect_derivative_rms'])
    rows.append(str(i)+' & '+' & '.join(f'{x:.4g}' for x in values)+r' \\')
rows += [r'\bottomrule',r'\end{tabular}']
(ROOT/'tables/validation.tex').write_text('\n'.join(rows)+'\n')
