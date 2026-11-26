import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.family":"serif","mathtext.fontset":"cm","font.size":8,
    "axes.labelsize":8,"axes.titlesize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"lines.linewidth":1.1,
    "xtick.direction":"in","ytick.direction":"in","figure.dpi":150,"savefig.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.03,"legend.frameon":False,
})
BLUE="#1f5fa8"; ORANGE="#d1601f"; GREEN="#2a8a4a"; GREY="#777777"; RED="#b3202a"
COL1=3.4; COL2=7.0
def panel(ax,label,**kw):
    ax.text(-0.14,1.04,label,transform=ax.transAxes,ha="left",va="bottom",fontsize=9,fontweight="bold")

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
