"""All-case analysis of the sealed selection policy and unused opportunities."""
from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;OUT=LAB.parent.parent/'outputs/neural-stimulus/update-policy'
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 run=LAB/'results/p61-learned-update-compatibility-001';s=json.loads((run/'summary.json').read_text());c=s['config'];train=json.loads((run/'training-summary.json').read_text());allcases=[];models=[]
 for model in s['models']:
  label=model['model'];decisions=model['decisions'];desc=json.loads((run/'models'/label/'targets.json').read_text());multi=np.array([q['degree']>=2 and q['distinct_lags']>=2 for q in desc])
  for case in range(6):
   byslot={j:[next(r for r in s['results'] if r['model']==label and r['case']==case and r['block']==b and r['slot']==j) for b in range(2)] for j in range(7)};base=byslot[0];br=np.array([r['r2'] for r in base]);old=np.min(br,0)>=.5;basecap=np.array([r['nonlinear_temporal_capacity'] for r in base]);records=[]
   for j in range(7):
    row=byslot[j];cap=np.array([r['nonlinear_temporal_capacity'] for r in row]);delta=cap-basecap;relative=delta/np.maximum(basecap,.25);tr=br.copy() if j==0 else np.array([r['transported_old_r2'] for r in row]);direct=np.array([r['direct_old_r2'] for r in row]);nr=np.array([r['r2'] for r in row]);retained=old&(tr.min(0)>=.5);raw=old&(direct.min(0)>=.5);bounded=all(r['bounded'] for r in row);joint=bool(j!=0 and bounded and all(r['bounded'] for r in base) and old.sum()>0 and min(relative)>=c['quality_acceptance'] and retained.sum()==old.sum());new=multi&(nr.min(0)>=.5)&(br.max(0)<.5)
    records.append(dict(slot=j,kind=row[0]['kind'],fraction=row[0]['fraction'],capacity=cap.tolist(),delta=delta.tolist(),relative_delta=relative.tolist(),bounded=bounded,eligible_old=int(old.sum()),retained_old=int(retained.sum()),direct_retained_old=int(raw.sum()),retained_ids=np.flatnonzero(retained).tolist(),lost_ids=np.flatnonzero(old&~retained).tolist(),all_old_retained=bool(old.sum()>0 and retained.sum()==old.sum()),joint_success=joint,new_reliable_nonlinear_ids=np.flatnonzero(new).tolist()))
   selected=decisions[case]['selected_slot'];oracle=[r for r in records[1:5] if r['joint_success']];best=max(oracle,key=lambda r:min(r['relative_delta'])) if oracle else records[0];allcases.append(dict(model=label,case=case,seed=decisions[case]['seed'],selected_slot=selected,baseline_capacity=basecap.tolist(),policy=records[selected],fixed=records[3],random=records[5],oracle=best,oracle_available=bool(oracle),candidates=records))
 for label in [*c['models'],'pooled']:
  cases=[r for r in allcases if label=='pooled' or r['model']==label];arms={}
  for arm in ['policy','fixed','random','oracle']:
   rows=[r[arm] for r in cases];arms[arm]=dict(mean_delta=np.mean([r['delta'] for r in rows],0).tolist(),mean_relative_delta=np.mean([r['relative_delta'] for r in rows],0).tolist(),updates=sum(r['slot']!=0 for r in rows),joint_successes=sum(r['joint_success'] for r in rows),complete_retention_cases=sum(r['all_old_retained'] for r in rows),nonempty_reference_cases=sum(r['eligible_old']>0 for r in rows),retained_old=sum(r['retained_old'] for r in rows),eligible_old=sum(r['eligible_old'] for r in rows),direct_retained_old=sum(r['direct_retained_old'] for r in rows),new_reliable_nonlinear_instances=sum(len(r['new_reliable_nonlinear_ids']) for r in rows))
  models.append(dict(model=label,charts=len(cases),oracle_opportunity_charts=sum(r['oracle_available'] for r in cases),arms=arms,policy_minus_fixed_mean_delta=np.mean([np.array(r['policy']['delta'])-r['fixed']['delta'] for r in cases],0).tolist()))
 pooled=models[-1]['arms']['policy'];positive=bool(pooled['updates'] and min(pooled['mean_delta'])>0 and pooled['joint_successes'])
 report=dict(question='Can learned neural-context selection transfer useful execution/interface pairs to fresh runtimes?',criterion_met=positive,aggregate=models,cases=allcases,scope='Twelve new runtime charts over two existing model checkpoints; one transferred acquisition cycle. Historical functional feedback trains selection; new-chart target labels are withheld until decisions are sealed. Oracle outcomes are retrospective opportunities, not choices available to the policy.')
 OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'analysis.json',report)
 ledger=dict(historical_training_parent=c['training_parent'],historical_native_work='P59/P60 and inherited P55/P56 costs are inherited and recorded in Checkpoint03 results/cost-ledger.json; no historical costs erased.',policy_training=train,models=s['models'],new_native_sequences=sum(m['native_work']['sequences'] for m in s['models']),selected_interface_bytes=14080,retained_policy_bytes=train['retained_policy_array_bytes'],deployment_native_calls_per_step=1,energy='not measured',accounting='All fresh chart acquisition, trials, source trajectories, unused candidates, controls and padding are charged. Independent native verification is added separately. Fixed+2 minimal acquisition uses one candidate while the selector source-measures four; matched experimental exposure is not a claim of equal minimal deployment acquisition cost.')
 dump(OUT/'cost-ledger.json',ledger)
 lines=['P61: learned execution-update and compatibility selection','',f'Prespecified positive-transfer criterion met: {positive}','']
 for model in models:
  lines.append(f"{model['model']}: {model['charts']} fresh charts; {model['oracle_opportunity_charts']} contain a joint-success acquired update")
  for arm,stats in model['arms'].items():lines.append(f"  {arm}: updates={stats['updates']}, mean score changes={stats['mean_delta']}, joint successes={stats['joint_successes']}, old programs={stats['retained_old']}/{stats['eligible_old']}, complete cases={stats['complete_retention_cases']}/{model['charts']}")
 lines+=['','All-case policy outcomes:']
 for r in allcases:
  p=r['policy'];lines.append(f"  {r['model']} seed{r['seed']}: fraction={p['fraction']:+.2f}, delta={p['delta']}, retention={p['retained_old']}/{p['eligible_old']}, joint={p['joint_success']}, oracle_available={r['oracle_available']}")
 lines+=['',report['scope']];(OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
