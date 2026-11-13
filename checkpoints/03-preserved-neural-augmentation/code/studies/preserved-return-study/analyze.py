"""P59 capability acquisition, repeated evidence and frozen-observer retention."""
from pathlib import Path
import json,time
import numpy as np
from capacity_engine import capacity_targets

HERE=Path(__file__).resolve().parent;LAB=HERE.parent;RUN=LAB/'results/p59-preserved-return-augmentation-001'
OUT=LAB.parent.parent/'outputs/neural-stimulus/preserved-augmentation';OUT.mkdir(parents=True,exist_ok=True)
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def key(r):return (r['model'],r['case'],r['kind'],r['fraction'])
def label(d):return '*'.join(f"P{degree}(u[t-{lag}])" for lag,degree in d['parts'])
def main():
    began=time.perf_counter();manifest=json.loads((RUN/'manifest.json').read_text());c=manifest['config']
    raw=[json.loads(l) for l in (RUN/'runs.jsonl').read_text().splitlines()];rows=[r for r in raw if r.get('event')=='assessment_result']
    completed=[m for m in ['0.6B','1.7B'] if (RUN/'models'/m/'model-summary.json').exists()]
    rows=[r for r in rows if r['model'] in completed];results=[]
    for model in completed:
        d=RUN/'models'/model;descriptors=json.loads((d/'targets.json').read_text());multi=np.array([t['degree']>=2 and t['distinct_lags']>=2 for t in descriptors])
        with np.load(d/'assessment.npz') as f:features=f['features'].copy();drive=f['drive'].copy()
        for case in range(len(features)):
            base=next(r for r in rows if r['model']==model and r['case']==case and r['kind']=='baseline')
            baseline=np.array([b['r2'] for b in base['blocks']]);old=np.min(baseline,axis=0)>=.5;robust_old=np.min(baseline,axis=0)>=.55
            head=np.load(d/f'readout-{case}-0.npz')
            for r in [r for r in rows if r['model']==model and r['case']==case]:
                a=np.array([b['r2'] for b in r['blocks']]);new=np.min(a,axis=0)>=.5;strict_gain=multi&new&(np.max(baseline,axis=0)<.5)
                robust_gain=multi&(np.min(a,axis=0)>=.55)&(np.max(baseline,axis=0)<=.45)
                frozen=[]
                for block in range(c['test_blocks']):
                    start=c['evaluation_washout']+c['evaluation_training']+block*c['evaluation_test'];Y,_=capacity_targets(drive,start,c['evaluation_test'])
                    X=features[case,r['slot'],start:start+c['evaluation_test']].astype(float)
                    pred=((X-head['feature_mean'])/head['feature_scale'])@head['readout']+head['target_mean'];rr=1-np.sum((pred-Y)**2,0)/np.sum((Y-Y.mean(0))**2,0)
                    if not r['bounded']:rr=np.full(len(descriptors),-1.)
                    frozen.append(rr)
                frozen=np.stack(frozen);frozen_new=np.min(frozen,axis=0)>=.5;lost=old&~frozen_new
                controlled=strict_gain.copy()
                if r['kind']=='acquired':
                    for kind in ['random_same_norm','random_same_action']:
                        ctrl=next(z for z in rows if z['model']==model and z['case']==case and z['kind']==kind and z['fraction']==r['fraction'])
                        rr=np.array([b['r2'] for b in ctrl['blocks']]);controlled&=np.max(rr,axis=0)<.5
                else:controlled[:]=False
                delta=[r['blocks'][j]['nonlinear_temporal_capacity']-base['blocks'][j]['nonlinear_temporal_capacity'] for j in range(2)]
                rec={k:v for k,v in r.items() if k not in ['event','blocks']};rec.update(
                    nonlinear_capacity=[b['nonlinear_temporal_capacity'] for b in r['blocks']],capacity_change=delta,capacity_improves_both_blocks=bool(min(delta)>0),
                    baseline_reliable_functions=int(old.sum()),baseline_reliable_nonlinear_functions=int(sum(old&multi)),
                    reliably_acquired_nonlinear_targets=np.flatnonzero(strict_gain).tolist(),
                    robustly_acquired_nonlinear_targets=np.flatnonzero(robust_gain).tolist(),
                    acquired_targets_beyond_both_controls=np.flatnonzero(controlled).tolist(),
                    refit_lost_old_targets=np.flatnonzero(old&~new).tolist(),
                    frozen_readout_lost_old_targets=np.flatnonzero(lost).tolist(),
                    frozen_readout_old_retained=int(sum(old&frozen_new)),frozen_readout_robust_old_lost=np.flatnonzero(robust_old&~frozen_new).tolist(),
                    retention_preserving_acquisition=bool(r['bounded'] and strict_gain.any() and not lost.any()),
                    nonvacuous_retention_preserving_acquisition=bool(r['bounded'] and old.any() and strict_gain.any() and not lost.any()),
                    controlled_retention_preserving_acquisition=bool(r['bounded'] and old.any() and controlled.any() and not lost.any()),
                    function_scores=dict(baseline=baseline.tolist(),acquired_readout=a.tolist(),frozen_baseline_readout=frozen.tolist()))
                results.append(rec)
    primary=[r for r in results if r['kind']=='acquired' and r['fraction']==c['primary_action_fraction']]
    arms={}
    for kind in ['acquired','random_same_norm','random_same_action']:
        selected=[r for r in results if r['kind']==kind and r['fraction']==c['primary_action_fraction']]
        arms[kind]=dict(cases=len(selected),bounded=sum(r['bounded'] for r in selected),
            reliable_gain_cases=sum(bool(r['reliably_acquired_nonlinear_targets']) for r in selected),
            nonvacuous_retention_preserving_gain_cases=sum(r['nonvacuous_retention_preserving_acquisition'] for r in selected),
            gained_target_instances=sum(len(r['reliably_acquired_nonlinear_targets']) for r in selected),
            frozen_readout_loss_instances=sum(len(r['frozen_readout_lost_old_targets']) for r in selected),
            capacity_improves_both_blocks=sum(r['capacity_improves_both_blocks'] for r in selected),
            mean_capacity_change_per_block=np.mean([r['capacity_change'] for r in selected],axis=0).tolist() if selected else [])
    report=dict(config=c,models_completed=completed,conditions=len(results),primary=arms,
        primary_controlled_retention_preserving_gain_cases=sum(r['controlled_retention_preserving_acquisition'] for r in primary),
        results=results,target_descriptors=descriptors if completed else [],analysis_seconds=time.perf_counter()-began,
        source_policy='Acquired directions and sign/action-RMS calibrated from neural responses; +2percent is prespecified primary; remaining doses map the response curve.',
        retention_scope='Frozen baseline observer measures retained executable functions; separately refitted candidate observers measure newly accessible computation.')
    dump(OUT/'analysis.json',report)
    concise={k:v for k,v in report.items() if k not in ['results','target_descriptors']};dump(OUT/'progress.json',concise);print(json.dumps(concise,indent=2))
    lines=['P59 PRESERVED-RETURN AUGMENTATION','Completed models: '+', '.join(completed),'',json.dumps(arms,indent=2),'','Per-case acquired dose results:']
    for r in results:
        if r['kind']=='acquired':lines.append(f"{r['model']} {r['parent'][:3]} seed{r['seed']} dose{r['fraction']:+.1%}: bounded={r['bounded']}; new={r['reliably_acquired_nonlinear_targets']}; frozen-old-loss={r['frozen_readout_lost_old_targets']}; capacity_delta={np.round(r['capacity_change'],4).tolist()}")
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
