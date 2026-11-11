"""Figure 2: revised RQ1 algorithm families, styled to match frozen 2026 artwork.
Run: python plot_figure2_reclassified.py
Optional: --input PATH --output-dir PATH
CPU-only; requires matplotlib, numpy, pandas and openpyxl.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

ROOT=Path(__file__).resolve().parent
FAMILIES=[
"Convolutional neural networks","Support vector machines / regression",
"Random forests","Transformers","Boosted-tree methods","Decision trees / rules",
"Logistic regression","Graph neural networks","MLP / feedforward neural networks",
"Clustering","Other specified computational methods","Other linear / regression models",
"Recurrent neural networks","k-nearest neighbors","Autoencoders",
"Discriminant analysis","Naive Bayes","Other specified neural architectures",
"Multiple models: no unique principal family","Neural networks: subtype unresolved",
"Principal family not resolved"]
COUNTS=[56,33,29,22,19,18,16,14,10,9,6,5,5,4,3,2,2,2,85,30,51]
STAGES=["Prescreening","Screening","Diagnosis","Prognosis","Monitoring/intervention","Not specified"]
LABELS=[
"CNN","SVM / SVR","Random forest","Transformer","Boosted trees",
"Decision tree / rules","Logistic regression","GNN","MLP / feedforward NN",
"Clustering","Other specified\ncomputational methods","Other linear / regression",
"Recurrent neural network","k-nearest neighbors","Autoencoder","Discriminant analysis",
"Naive Bayes","Other specified\nneural architectures",
"Multiple models /\nno unique family","Neural network /\nsubtype unresolved",
"Principal family\nnot resolved"]
DOI_SHA256="c0391238635a7c8e8c1a4608b1cb72d8a694b658774e326f0908d71df1055a8d"

def load_inventory(source,allocation):
    if source.suffix.lower()==".csv":
        df=pd.read_csv(source,dtype={"Report ID":"string"})
    else:
        df=pd.read_excel(source,sheet_name="02_Included_reports",header=1,dtype={"Report ID":"string"})
    df=df.loc[df["Report ID"].fillna("").str.fullmatch(r"S\d{3}")].copy()
    if len(df)!=421 or df["Report ID"].tolist()!=[f"S{i:03}" for i in range(1,422)]:
        raise ValueError("Expected S001-S421 exactly once and in order")
    dois=(df["DOI"].fillna("").astype(str).str.strip().str.lower()
          .str.replace(r"^https?://(dx\.)?doi\.org/","",regex=True))
    digest=hashlib.sha256(("\n".join(dois)+"\n").encode()).hexdigest()
    if digest!=DOI_SHA256: raise ValueError("DOI inventory mismatch: "+digest)
    if set(df["Primary functional stage"])-set(STAGES):
        raise ValueError("Unexpected functional stages")
    overlay=pd.read_csv(allocation,dtype="string")
    if len(overlay)!=421 or overlay["Report ID"].tolist()!=df["Report ID"].tolist():
        raise ValueError("RQ1 overlay ID mismatch")
    if set(overlay["Revised allocation"])!=set(FAMILIES):
        raise ValueError("Unknown or missing family")
    df["Revised allocation"]=overlay["Revised allocation"].to_numpy()
    observed=[int((df["Revised allocation"]==f).sum()) for f in FAMILIES]
    if observed!=COUNTS: raise ValueError("Family totals differ from audited baseline")
    return df

def render(df,out):
    if out.resolve()==(ROOT/"figures_manuscript_421").resolve():
        raise ValueError("Frozen manuscript figures must not be overwritten")
    out.mkdir(parents=True,exist_ok=True)
    table=(df.groupby(["Revised allocation","Primary functional stage"]).size()
           .unstack(fill_value=0).reindex(index=FAMILIES,columns=STAGES,fill_value=0))
    if table.to_numpy().sum()!=421: raise AssertionError("421 reports not conserved")
    long=table.stack().rename("Reports").reset_index()
    long.columns=["Algorithm_family","Primary_functional_stage","Reports"]
    long.to_csv(out/"Figure_02_family_stage_counts.csv",index=False)
    long.to_csv(out/"figure_02_algorithm_stage.csv",index=False)
    df[["Report ID","DOI","Primary functional stage","Revised allocation"]].to_csv(
        out/"Figure_02_report_allocation.csv",index=False)
    plt.rcParams.update({"font.family":"Liberation Serif","font.size":8.6,
        "svg.fonttype":"none","pdf.fonttype":42,"figure.facecolor":"white"})
    fig,ax=plt.subplots(figsize=(6.86,7.2))
    fig.subplots_adjust(left=.352,right=.99,bottom=.085,top=.986)
    norm=Normalize(vmin=0,vmax=max(1,int(table.to_numpy().max())))
    cmap=plt.get_cmap("viridis")
    for y,family in enumerate(FAMILIES):
        for x,stage in enumerate(STAGES):
            n=int(table.loc[family,stage])
            if not n: continue
            ax.scatter(x,y,s=46+365*(n/85)**.5,color=[cmap(norm(n))],
                       edgecolors="white",linewidths=.45,zorder=3)
            ax.text(x,y,str(n),fontsize=8.4,ha="center",va="center",
                    color="#122127" if norm(n)>=.70 else "white",zorder=4)
    ax.set_xticks(range(6),["Prescreening","Screening","Diagnosis","Prognosis",
                              "Monitoring/\nintervention","Not specified"],fontsize=8.5)
    ax.set_yticks(range(21),LABELS,fontsize=8.5)
    ax.set_xlim(-.48,5.52);ax.set_ylim(20.55,-.48)
    ax.grid(axis="both",color="#e5e7eb",linewidth=.6);ax.set_axisbelow(True)
    ax.axhline(17.5,color="#9da9b3",linewidth=.75,linestyle=(0,(3,3)),zorder=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlabel("Primary functional stage",fontsize=9.5,labelpad=7)
    ax.set_ylabel("Algorithm family / identification status",fontsize=9.5,labelpad=12)
    for ext in ("png","pdf","svg"):
        fig.savefig(out/("Figure_02."+ext),
                    dpi=330 if ext=="png" else None,facecolor="white")
    plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"Supplementary_Data_S1_simplificado_421.xlsx")
    p.add_argument("--allocation",type=Path,default=ROOT/"rq1_algorithm_family_421.csv")
    p.add_argument("--output-dir",type=Path,default=ROOT/"figures_revised_421")
    a=p.parse_args(argv)
    render(load_inventory(a.input,a.allocation),a.output_dir)
    print("Revised Figure 2 generated; all 421 reports included.")

if __name__=="__main__":main()
