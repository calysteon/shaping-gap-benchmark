from pathlib import Path
import json
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'plot-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).parent/'plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CP=Path(__file__).resolve().parents[2]
PAPER=CP/'paper/source'
s=json.loads((CP/'results/summary.json').read_text())
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(6.5,2.55),layout='constrained')
for m,label,color in [('native_local','Native local','#54637b'),('factored_pair','Acquired pair interface','#087e8b')]:
 r=[v for v in s['scaling'] if v['method']==m];x=[v['bank_size'] for v in r]
 axs[0].plot(x,[v['solved'] for v in r],'o-',label=label,color=color)
 axs[1].plot(x,[v['with_selected_program_verification_lpe']/1000 for v in r],'o-',color=color)
axs[0].set(ylabel='Contracts satisfied / 72',ylim=(0,76));axs[0].legend(fontsize=7,loc='lower right')
axs[1].set(ylabel='Charged native work (thousand LPE)')
for a in axs:a.set_xlabel('Available frozen components');a.set_xticks([22,128,256,512]);a.grid(alpha=.15)
fig.savefig(PAPER/'figures/scaling.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(6.5,2.55),layout='constrained')
for key,label,col in [('direct_median_seconds','Direct states','#b85f18'),('factored_median_seconds','Response moments','#087e8b')]:
 axs[0].plot([r['bank_size'] for r in s['micro']],[r[key]*1000 for r in s['micro']],'o-',label=label,color=col)
axs[0].set(ylabel='Median forecast ms (log)',yscale='log');axs[0].legend(fontsize=7)
r=[v for v in s['scaling'] if v['method']=='factored_pair'];x=[v['bank_size'] for v in r]
axs[1].plot(x,[v['retained_bank_bytes']/2**20 for v in r],'o-',label='Retained arrays')
axs[1].plot(x,[v['controller_array_bound_bytes']/2**20 for v in r],'o-',label='Controller reservation')
axs[1].axhline(64,color='gray',ls='--',label='Array limit');axs[1].set(ylabel='MiB',ylim=(0,70));axs[1].legend(fontsize=7)
for a in axs:a.set_xlabel('Available frozen components');a.set_xticks([22,128,256,512]);a.grid(alpha=.15)
fig.savefig(PAPER/'figures/resources.pdf');plt.close(fig)
runs=json.loads((CP/'data/functions/summary.json').read_text())['results']
rows=[]
for name in ['and2','or2','xor2','and3','or3','xor3','majority3','mux3']:
 r=[v for v in s['by_function'] if v['function']==name];a,b=r
 rows.append(f"{name.replace('_',r'\_')} & {a['exact']}/3 & {b['exact']}/3 & {a['robust']}/3 & {b['robust']}/3 & {a['construction_lpe']:,} & {b['construction_lpe']:,} \\")
(PAPER/'function-rows.tex').write_text('\n'.join(rows)+'\n')
example={r['method']:r for r in runs if r['function']=='xor3' and r['seed']==79003}
npz=np.load(CP/'data/functions/cases'/f"t{example['paired']['case']}.npz")
rows=[]
for i,bits in enumerate(npz['bits']):
 rows.append(' & '.join(str(int(v)) for v in bits)+f" & {int(npz['labels'][i])} & {example['one_step']['final_q'][i][0]:.6f} & {example['paired']['final_q'][i][0]:.6f} \\")
(PAPER/'example-rows.tex').write_text('\n'.join(rows)+'\n')
(PAPER/'example-program.tex').write_text(', '.join(str(a['unit_index']) for a in example['paired']['graph']))
for name in ['function-rows.tex','example-rows.tex']:
 p=PAPER/name;p.write_text('\n'.join(line+chr(92) for line in p.read_text().splitlines())+'\n')
print('Generated two figures, per-function table and an exhaustive native XOR example.')
