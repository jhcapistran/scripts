"""Recompute AI-ASD manuscript Figures 2-8 from the final 421-report supplement.

The **exact artwork already inserted in the Word manuscript** is versioned separately
under ``figures_manuscript_421``. This script makes a fresh, data-derived rendering (including revised RQ1 allocation) in
``figures_recomputed_421`` and exports the aggregate tables for independent auditing.

It deliberately does not regenerate Figure 1 (PRISMA), because the historical search
flow is not yet reconciled. No data are inferred for unknown charting fields.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / 'Supplementary_Data_S1_simplificado_421.xlsx'
OUTPUT = BASE_DIR / 'figures_recomputed_421'
FIGURES = (2, 3, 4, 5, 6, 7, 8)
STAGES = ('Prescreening', 'Screening', 'Diagnosis', 'Prognosis', 'Monitoring/intervention', 'Not specified')
METHODS = ('Machine Learning', 'Deep Learning', 'Hybrid', 'Not specified')
SOURCES = (
    'Neuroimaging', 'Structured clinical/questionnaire data',
    'Behavioral images/video/gaze/movement', 'Physiological signals',
    'Multiple source categories', 'Biological/omics', 'Text/language',
    'Not specified', 'Audio/voice', 'Other biomedical/digital data',
)
ALGORITHMS = (
    'Other specified method', 'Multiple/not uniquely specified',
    'Composite/ensemble pipeline', 'CNN', 'SVM', 'Random forest', 'Transformer',
    'Decision tree/rules', 'GNN', 'Logistic regression', 'Gradient boosting',
    'Clustering', 'MLP/ANN', 'Recurrent network', 'Autoencoder', 'k-nearest neighbors',
)
PRACTICES = (
    ('External validation', 'External validation'),
    ('Multisource integration', 'Multisource integration'),
    ('Explainability: broad', 'Explainability (broad)'),
    ('Cross-site or cross-dataset evaluation', 'Cross-site / cross-dataset evaluation'),
)
REPORTING = ('Reported', 'Not reported in assessed sources', 'Not ascertainable')
TIMINGS = ('Pre-decision', 'In-decision', 'Post-decision')
MATURITY = ('Research only', 'Proposed only', 'Evaluated AI use', 'Not ascertainable')
EVALUATED = {'Evaluated at point of use', 'Evaluated AI-supported intervention', 'Evaluated care-delivery platform'}
MATURITY_COLORS = {'Research only': '#b8c0c8', 'Proposed only': '#0072b2', 'Evaluated AI use': '#d55e00', 'Not ascertainable': '#e69f00'}
TIMING_COLORS = ('#0072b2', '#e69f00', '#009e73')
REPORTING_COLORS = ('#2a7f7b', '#c9ced3', '#c79a3b')


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError('The 421-report supplement is required: ' + str(path))
    # Title occupies row 1 and column headings row 2.
    df = pd.read_excel(path, sheet_name='02_Included_reports', header=1)
    df = df.loc[df['Report ID'].astype(str).str.match(r'^S\d{3}$')].copy()
    df = df.fillna('Not ascertainable')
    if len(df) != 421:
        raise ValueError(f'Expected 421 included reports, got {len(df)}')
    expected = [f'S{i:03d}' for i in range(1, 422)]
    if df['Report ID'].tolist() != expected:
        raise ValueError('Report IDs are not the consecutive S001-S421 inventory')
    doi = df['DOI'].astype(str).str.lower().str.strip().str.replace(r'^https?://(dx\.)?doi\.org/', '', regex=True)
    if doi.eq('not ascertainable').any() or doi.duplicated().any():
        raise ValueError('Missing or duplicate DOI found')
    for field, valid in [('Primary functional stage', STAGES), ('AI category', METHODS), ('Primary data source', SOURCES)]:
        unknown = set(df[field].astype(str)) - set(valid)
        if unknown:
            raise ValueError(f'Unexpected categories for {field}: {unknown}')
    df['Maturity'] = df['Integration status'].apply(lambda s: 'Evaluated AI use' if s in EVALUATED else s)
    if set(df['Maturity']) - set(MATURITY):
        raise ValueError('Unexpected integration status detected')
    expected_counts = {'Research only': 307, 'Proposed only': 95, 'Evaluated AI use': 18, 'Not ascertainable': 1}
    if df['Maturity'].value_counts().to_dict() != expected_counts:
        raise ValueError('Integration counts differ from updated Word: ' + str(df['Maturity'].value_counts().to_dict()))
    for field,_ in PRACTICES:
        unexpected = set(df[field]) - set(REPORTING)
        if unexpected:
            raise ValueError(f'Unexpected reporting status for {field}: {unexpected}')
    return df


def profiles(df):
    x=df[(df['Primary functional stage']!='Not specified') & (df['Primary data source']!='Not specified') & (df['AI category']!='Not specified')]
    counts=x.groupby(['Primary functional stage','Primary data source','AI category']).size().to_dict()
    retained = {k:v for k,v in counts.items() if v>=8}
    stages={k[0] for k in retained}
    for s in STAGES[:-1]:
        if s not in stages:
            choices=[(k,v) for k,v in counts.items() if k[0]==s]
            if choices:
                k,v=max(choices,key=lambda q:(q[1],q[0][1],q[0][2]))
                retained[k]=v
    # Manuscript figure layout uses stage grouping, descending profile size.
    result=sorted(retained.items(),key=lambda q:(STAGES.index(q[0][0]),-q[1],q[0][1],q[0][2]))
    if len(result)!=17 or sum(n for _,n in result)!=275:
        raise ValueError(f'Figure 8 must show 17 profiles covering 275 reports, not {len(result)}/{sum(n for _,n in result)}')
    return result


def save_all(fig, directory:Path, name:str):
    directory.mkdir(parents=True,exist_ok=True)
    for extension in ('.png','.pdf','.svg'):
        p=directory/(name+extension)
        fig.savefig(p, bbox_inches='tight', dpi=350 if extension=='.png' else None)
    plt.close(fig)


def simplify_source(s):
    return {'Structured clinical/questionnaire data':'Clinical/questionnaire', 'Behavioral images/video/gaze/movement':'Behavioral/gaze/movement', 'Multiple source categories':'Multiple sources', 'Physiological signals':'Physiological signals', 'Other biomedical/digital data':'Other biomedical/digital'}.get(s,s)


def simplify_stage(s):
    return 'Monitoring/\nintervention' if s=='Monitoring/intervention' else s


def draw_figure2(df, out):
    """Reclassified 421-report RQ1 figure, matched to the original bubble plot.

    Uses an explicit, independently auditable per-report allocation CSV.
    Never changes or rewrites the frozen embedded manuscript assets.
    """
    from plot_figure2_reclassified import render, FAMILIES, COUNTS
    mapping = pd.read_csv(BASE_DIR / 'rq1_algorithm_family_421.csv', dtype='string')
    if len(mapping) != 421 or mapping['Report ID'].tolist() != df['Report ID'].tolist():
        raise ValueError('RQ1 revised classification must match all 421 report IDs')
    if set(mapping['Revised allocation']) != set(FAMILIES):
        raise ValueError('Unexpected family in RQ1 revised allocation')
    work = df.copy()
    work['Revised allocation'] = mapping['Revised allocation'].to_numpy()
    if [int((work['Revised allocation'] == f).sum()) for f in FAMILIES] != COUNTS:
        raise ValueError('RQ1 category totals differ from audited 421-report allocation')
    render(work, out)


def draw_figure3(df,out):
    counter=Counter(zip(df['Primary functional stage'],df['AI category'],df['Primary data source']))
    rows=[]
    for s in STAGES:
        for m in METHODS:
            for src in SOURCES:
                rows.append((s,m,src,counter.get((s,m,src),0)))
    pd.DataFrame(rows,columns=['Stage','AI_category','Data_source','Reports']).to_csv(out/'figure_03_method_source_stage.csv',index=False)
    matrix=np.array([[counter.get((s,m,src),0) for s in STAGES for m in METHODS] for src in SOURCES],dtype=float)
    fig,ax=plt.subplots(figsize=(18.5,6.7))
    im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,aspect='auto')
    for y in range(len(SOURCES)):
        for x in range(len(STAGES)*len(METHODS)):
            n=int(matrix[y,x]); ax.text(x,y,str(n),ha='center',va='center',fontsize=6.8,color='white' if n>=19 else '#525454')
    ax.set_yticks(np.arange(len(SOURCES)),[simplify_source(x) for x in SOURCES],fontsize=9)
    ax.set_xticks(np.arange(matrix.shape[1]),[{'Machine Learning':'Machine\nLearning','Deep Learning':'Deep\nLearning','Hybrid':'Hybrid','Not specified':'Not specified'}[m] for s in STAGES for m in METHODS],fontsize=7,rotation=65,ha='right')
    for i,s in enumerate(STAGES):
        ax.text(4*i+1.5,-1.05,s.replace('/','/\n') if s=='Monitoring/intervention' else s,ha='center',va='center',fontsize=9)
        if i: ax.axvline(i*4-.5,color='white',linewidth=2)
    ax.set_ylabel('Primary data source');ax.set_xlabel('AI category grouped by primary functional stage',labelpad=8)
    ax.set_ylim(len(SOURCES)-.5,-1.5)
    for sp in ax.spines.values():sp.set_visible(False)
    fig.colorbar(im,ax=ax,shrink=.88,pad=.015,label='Reports')
    fig.tight_layout();save_all(fig,out,'Figure_03')


def plot_stacked(ax,labels,counts,order,colors,denominators,label_threshold=10):
    left=np.zeros(len(labels),dtype=float)
    for key in order:
        vals=np.array([counts.get((label,key),0) for label in labels],dtype=float)
        widths=np.divide(vals,denominators,out=np.zeros_like(vals),where=denominators!=0)*100
        b=ax.barh(np.arange(len(labels)),widths,left=left,color=colors[key] if isinstance(colors,dict) else colors[list(order).index(key)],height=.55,edgecolor='white',linewidth=.65,label=key)
        for i,rect in enumerate(b):
            if vals[i]==0: continue
            text=f'{int(vals[i])} ({widths[i]:.1f}%)'
            if widths[i]>=label_threshold:
                shade=colors[key] if isinstance(colors,dict) else colors[list(order).index(key)]
                c='white' if shade.lower() in {'#0072b2','#d55e00','#009e73','#2a7f7b'} else '#26323b'
                ax.text(left[i]+widths[i]/2,i,text,ha='center',va='center',fontsize=8,color=c)
            elif widths[i]>=2:
                ax.text(left[i]+widths[i]/2,i,str(int(vals[i])),ha='center',va='center',fontsize=8,color='#20202a')
        left+=widths
    ax.set_yticks(range(len(labels)),labels); ax.invert_yaxis()
    ax.set_xlim(0,112); ax.set_xticks([0,25,50,75,100],[f'{x}%' for x in [0,25,50,75,100]])
    ax.grid(axis='x',color='#e4e8ed',linewidth=.5); ax.set_axisbelow(True)
    for sp in ax.spines.values():sp.set_visible(False)


def draw_figure4(df,out):
    counts=Counter(zip(df['Primary functional stage'],df['Maturity']))
    stages=list(STAGES)
    pd.DataFrame([(s,m,counts.get((s,m),0)) for s in STAGES for m in MATURITY],columns=['Stage','Maturity','Reports']).to_csv(out/'figure_04_integration_stage.csv',index=False)
    totals=np.array([sum(counts.get((s,m),0) for m in MATURITY) for s in stages],dtype=float)
    fig,ax=plt.subplots(figsize=(10.8,6))
    labels=[f'{simplify_stage(s).replace(chr(10)," ")} (n={int(n)})' for s,n in zip(stages,totals)]
    plot_stacked(ax,labels,{(labels[i],m):counts.get((s,m),0) for i,s in enumerate(stages) for m in MATURITY},MATURITY,MATURITY_COLORS,totals)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.2),frameon=False,ncol=2,fontsize=9)
    ax.set_xlabel('Within-stage proportion (%)');fig.tight_layout();save_all(fig,out,'Figure_04')


def draw_figure5(df,out):
    sub=df.loc[df['Maturity'].isin(('Proposed only','Evaluated AI use'))]
    groups=['Proposed only','Evaluated AI use']
    counts=Counter()
    for _,r in sub.iterrows():
        status=r['Maturity'];col='Intended decision timing' if status=='Proposed only' else 'Observed decision timing'
        counts[(status,str(r[col]))]+=1
    if counts.get(('Proposed only','Not ascertainable'),0) or counts.get(('Evaluated AI use','Not ascertainable'),0):
        raise ValueError('Unexpected missing timing; Figure 5 denominators require 95 and 18 classified')
    vals=[sum(counts.get((g,t),0) for t in TIMINGS) for g in groups]
    if vals!=[95,18]:raise ValueError('Figure 5 expected n=95 and n=18, got '+str(vals))
    pd.DataFrame([(g,t,counts.get((g,t),0)) for g in groups for t in TIMINGS],columns=['Maturity','Timing','Reports']).to_csv(out/'figure_05_decision_timing.csv',index=False)
    fig,ax=plt.subplots(figsize=(10.6,3.4))
    lbl=[f'{g} (n={n})' for g,n in zip(groups,vals)]
    c={(lbl[i],t):counts.get((g,t),0) for i,g in enumerate(groups) for t in TIMINGS}
    plot_stacked(ax,lbl,c,TIMINGS,TIMING_COLORS,np.array(vals),label_threshold=9)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.38),frameon=False,ncol=3,fontsize=9)
    ax.set_xlabel('Within-group proportion (%)');fig.tight_layout();save_all(fig,out,'Figure_05')


def draw_figure6(df,out):
    n=len(df)
    counts={(lbl,s):int((df[field]==s).sum()) for field,lbl in PRACTICES for s in REPORTING}
    pd.DataFrame([(lbl,s,counts[(lbl,s)]) for field,lbl in PRACTICES for s in REPORTING],columns=['Practice','Reporting_status','Reports']).to_csv(out/'figure_06_practice_reporting.csv',index=False)
    labels=[lbl for _,lbl in PRACTICES]
    fig,ax=plt.subplots(figsize=(11.4,4.5))
    plot_stacked(ax,labels,counts,REPORTING,REPORTING_COLORS,np.array([n]*len(labels)),label_threshold=9)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.45),frameon=False,ncol=3,fontsize=8.5)
    ax.set_xlabel(f'Share of reports (N={n})');fig.tight_layout();save_all(fig,out,'Figure_06')


def draw_figure7(df,out):
    rows=[]
    for s in STAGES:
        subset=df[df['Primary functional stage']==s]
        for col,lbl in PRACTICES:
            rows.append((s,lbl,len(subset),int((subset[col]=='Reported').sum())))
    pd.DataFrame(rows,columns=['Stage','Practice','Denominator','Reported']).to_csv(out/'figure_07_practice_by_stage.csv',index=False)
    rates=np.array([[n/den if den else 0 for _,l,den,n in rows if l==lbl] for _,lbl in PRACTICES]).T
    fig,ax=plt.subplots(figsize=(10.8,5.4))
    im=ax.imshow(rates,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i,s in enumerate(STAGES):
        for j,(_,lbl) in enumerate(PRACTICES):
            _,_,den,n=rows[i*len(PRACTICES)+j]
            ax.text(j,i,f'{n}/{den}',ha='center',va='center',fontsize=10,color='white' if rates[i,j]>.55 else '#273849')
    ax.set_xticks(range(4),['External\nvalidation','Multisource\nintegration','Explainability\n(broad)','Cross-site /\ncross-dataset\nevaluation'],fontsize=10)
    ax.xaxis.tick_top();ax.set_yticks(range(6),[f'{simplify_stage(s).replace(chr(10)," ")} (n={len(df[df["Primary functional stage"]==s])})' for s in STAGES],fontsize=10)
    for sp in ax.spines.values():sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.15)
    cb.set_label('Reports documenting the practice (%)');cb.set_ticks([0,.25,.5,.75,1]);cb.set_ticklabels(['0%','25%','50%','75%','100%'])
    fig.tight_layout();save_all(fig,out,'Figure_07')


def draw_figure8(df,out):
    ps=profiles(df)
    result=[]
    for (s,source,cat),den in ps:
        subset=df[(df['Primary functional stage']==s)&(df['Primary data source']==source)&(df['AI category']==cat)]
        for field,label in PRACTICES:
            result.append({'Stage':s,'Data_source':source,'AI_category':cat,'Denominator':den,'Practice':label,'Reported':int((subset[field]=='Reported').sum())})
    pd.DataFrame(result).to_csv(out/'figure_08_selected_profiles.csv',index=False)
    matrix=np.array([[row['Reported']/row['Denominator'] for row in result[i*4:i*4+4]] for i in range(len(ps))])
    fig,ax=plt.subplots(figsize=(10.5,11.0))
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    labels=[]
    short={'Machine Learning':'ML','Deep Learning':'DL','Hybrid':'Hybrid'}
    for ((s,source,cat),den) in ps:
        st=s if s in ('Prescreening','Screening','Diagnosis','Prognosis') else s.replace('/','/')
        labels.append(f'{st}: {simplify_source(source)} - {short.get(cat,cat)} (n={den})')
    ax.set_yticks(range(len(ps)),labels,fontsize=8)
    ax.set_xticks(range(4),['External\nvalidation','Multisource\nintegration','Explainability\n(broad)','Cross-site /\ncross-dataset\nevaluation'],fontsize=10)
    ax.xaxis.tick_top()
    for i in range(len(ps)):
        for j in range(4):
            r=result[i*4+j]; ax.text(j,i,f'{r["Reported"]}/{r["Denominator"]}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.57 else '#34404c')
    for i in range(1,len(ps)):
        if ps[i][0][0]!=ps[i-1][0][0]:ax.axhline(i-.5,color='white',linewidth=2)
    for sp in ax.spines.values():sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.06,pad=.07)
    cb.set_ticks([0,.25,.5,.75,1]);cb.set_ticklabels(['0%','25%','50%','75%','100%']);cb.set_label('Reports documenting the practice (%)')
    fig.tight_layout();save_all(fig,out,'Figure_08')


def figure_table(df,out):
    subset=df[df['Maturity'].isin(['Proposed only','Evaluated AI use'])]
    roles=subset.groupby('Functional role').size().sort_values(ascending=False).rename('Reports')
    roles.to_csv(out/'table_03_functional_roles.csv',header=True)
    return {'N':len(df), 'stage_counts':df['Primary functional stage'].value_counts().to_dict(), 'method_counts':df['AI category'].value_counts().to_dict(),'integration_counts':df['Maturity'].value_counts().to_dict(),'evaluated_subcategories':df.loc[df['Maturity']=='Evaluated AI use','Integration status'].value_counts().to_dict(),'practices_reported':{lbl:int((df[f]=='Reported').sum()) for f,lbl in PRACTICES},'selected_profiles':{'n_profiles':len(profiles(df)),'reports':sum(n for _,n in profiles(df))},'proposed_evaluated_n':int(len(subset))}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=SOURCE,help='421-report Supplementary Data S1 (.xlsx)')
    p.add_argument('--output-dir',type=Path,default=OUTPUT,help='Generated outputs; never overwrites frozen Word graphics')
    p.add_argument('--figures',nargs='*',type=int,default=list(FIGURES),help='Subset from Figures 2..8')
    p.add_argument('--check-only',action='store_true',help='Validate source and print counts without rendering')
    a=p.parse_args(argv)
    df=load_data(a.input)
    if not a.check_only and a.output_dir.resolve()==(BASE_DIR/'figures_manuscript_421').resolve():
        raise ValueError('Refusing to overwrite the exact figures extracted from the Word manuscript.')
    if not a.check_only: a.output_dir.mkdir(parents=True,exist_ok=True)
    summary=figure_table(df,a.output_dir) if not a.check_only else {'N':len(df),'stages':df['Primary functional stage'].value_counts().to_dict(),'maturity':df['Maturity'].value_counts().to_dict(),'profile_count':len(profiles(df))}
    if a.check_only:
        print(json.dumps(summary,indent=2,ensure_ascii=False));return
    a.output_dir.mkdir(parents=True,exist_ok=True)
    methods={2:draw_figure2,3:draw_figure3,4:draw_figure4,5:draw_figure5,6:draw_figure6,7:draw_figure7,8:draw_figure8}
    for num in a.figures:
        if num not in methods:raise ValueError(f'Unsupported Figure {num}: only Figures 2-8')
        methods[num](df,a.output_dir)
        print(f'Figure {num} rendered from {len(df)} reports')
    (a.output_dir/'summary_421.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Completed. PRISMA/Figure 1 intentionally untouched.')

if __name__=='__main__':main()
