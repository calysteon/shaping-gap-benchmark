"""Conceptual dataflow for the implemented acquisition/construction runtime."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(11.2,4.4));fig.subplots_adjust(left=.01,right=.99,bottom=.015,top=.99)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
        lw=1.0,edgecolor=color,facecolor=color+'12'))
    ax.text(x+w/2,y+h-.047,title,ha='center',va='top',fontsize=12.5,weight='bold',color='#19313D')
    ax.text(x+w/2,y+.055,body,ha='center',va='bottom',fontsize=11.8,linespacing=1.4,color='#19313D')

def arrow(a,b,color='#3C6773',style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
        linewidth=1.25,color=color,linestyle=style,connectionstyle='arc3,rad=0'))

ax.text(.015,.96,'ACQUISITION',fontsize=10.5,weight='bold',color='#366A63')
ax.text(.985,.96,'Fixed source domains; descriptions sealed before later contracts',ha='right',fontsize=10.3,color='#52646D')
box(.02,.63,.235,.245,'Frozen native bank','Original components\nFixed parameters','#5B7480')
box(.345,.63,.275,.245,'Source observations','Amplitudes and effects\nIncoming-context partials','#267C75')
box(.705,.63,.275,.245,'Retained interfaces','Behavior across contexts\nReused across contracts','#267C75')
arrow((.267,.752),(.335,.752));arrow((.632,.752),(.695,.752))

ax.text(.015,.44,'CONSTRUCTION',fontsize=10.5,weight='bold',color='#356187')
box(.02,.10,.205,.245,'Requirement','Live state\nTarget/protected intervals','#4F789B')
box(.285,.10,.215,.245,'Compatibility','Forecast continuations\nRank completion or reuse','#4F789B')
box(.56,.10,.205,.245,'Native branches','Recompute live activity\nCheck actual constraints','#4F789B')
box(.825,.10,.155,.245,'Deployment','Selected program\nNative witness','#4F789B')
arrow((.237,.222),(.275,.222));arrow((.512,.222),(.55,.222));arrow((.777,.222),(.815,.222))
arrow((.84,.617),(.405,.358),'#267C75')
ax.text(.66,.50,'Query retained behavior',ha='center',fontsize=10.3,color='#267C75',rotation=0,
        bbox={'facecolor':'white','edgecolor':'none','pad':1})
ax.annotate('',xy=(.39,.067),xytext=(.66,.067),
    arrowprops=dict(arrowstyle='-|>',color='#637C89',lw=1.0,connectionstyle='arc3,rad=0'))
ax.text(.53,.005,'Observed state updates the search frontier',ha='center',va='bottom',fontsize=10.3,color='#52646D')

for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/f'interface-runtime.{ext}',dpi=180,facecolor='white')
plt.close(fig)
