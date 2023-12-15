
import os
import warnings
from pathlib import Path

import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc,ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import *

# Directory of font files
# curretly there are **Liberation** and **MathJax** fonts
# Arial is replaced with **Liberation** because Arial is not a free font.
# import misc
# font_dirs = [os.path.abspath(list(misc.__path__)[0]+'/fonts/'), ]
font_dirs = [os.path.abspath('./misc/fonts/'), ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# Deprecated:
# font_list = font_manager.createFontList(font_files) # createfontList is deprecated
# font_manager.fontManager.ttflist.extend(font_list)
for i in font_files:
    font_manager.fontManager.addfont(i)

# A commonly used rectangular for tight layout
rect = [0,0,1,0.95]
mono={'family' : 'monospace'}

def scatter(*arg,**kwargs):
    """Set the rasterized of scatter plot default to True unless specified in scatter()
    """
    if "rasterized" not in kwargs:
        kwargs["rasterized"]=True
    plt.scatter(*arg,**kwargs)


def legend_top(ncol=2, columnspacing = 0.5,bbox_to_anchor=(0., 1.02, 1., .102),*args, **kwargs):
    """
    Put legend on top of axis
    """
    legend(ncol=ncol, columnspacing = columnspacing,handlelength=1.,bbox_to_anchor=bbox_to_anchor,
    mode="expand", borderaxespad=0.,prop={'size': 14}, loc='lower left', frameon=False)   
    
def legend_top_figure(ncol=4, columnspacing = 0.5, bbox_to_anchor=(0.05, 0.95, 0.9, 0.95)):
    """
    Put legend on top of figure
    """    
    handles, labels = gca().get_legend_handles_labels()
    lgd=gcf().legend(ncol=ncol, columnspacing = columnspacing, handlelength=1.,bbox_to_anchor=bbox_to_anchor,
        mode="expand", borderaxespad=0.,prop={'size': 14}, loc='lower left', frameon=False) 
    return lgd
    
def ttext(x,y,info_to_display,mono=True,**kwargs):
    """
    A wrapper of plt.text, with x,y default to percentage of axis instead
    (transform=gca().transAxes)
    """
    if mono:
        mono={'family' : 'monospace'}
        text(x,y,info_to_display,transform=gca().transAxes,fontdict=mono,**kwargs)
    else:
        text(x,y,info_to_display,transform=gca().transAxes,**kwargs)


def plt_config(family="serif",usetex=False,math_fontfamily=None, fontsize_multi=1):
    """
    Figure presets. Changing the default rcParams.
    
    input:
    ---
    family: {"serif","sans-serif", or an exact font name}
        By default, "serif" will use `Liberation Sans` (a non comercial version of Arial) for normal text and dejavusans for mathtext.
        "sans-serif" will use `MathJax_Main` for normal text and cm for math text
        You can also directly provide a font name. To get a preview of what font you have, run show_all_fonts()
        Note that if you provide a font name, it will not affect the math text font. You need to set it with math_fontfamily
    usetex: bool
    math_fontfamily: str, the font name to be used by mathtext
    fontsize_multi: int
        Scaling the font size. All fontsize will be multiplied by this number. Default is 1.
    """
    
    # Set font type
    if family=="serif":
        plt.rcParams['font.family'] = 'MathJax_Main'
        plt.rcParams['mathtext.fontset'] = 'cm'
    elif family=="sans-serif":
        plt.rcParams['font.family'] = 'Liberation Sans'
        plt.rcParams['mathtext.fontset'] = 'dejavusans' 
    else:
        plt.rcParams['font.family'] = family
    
    if type(math_fontfamily) is str:
        plt.rcParams['mathtext.fontset'] = math_fontfamily  
        
    # Toggle Latex    
    if usetex:
        rc('text', usetex=False)
        mpl.rcParams['text.latex.preamble'] = [
              r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
              r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
               r'\usepackage{cms10}',    # set the normal font here
               r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
               r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
        ]

    # Font size
    plt.rcParams['font.size']      =14*fontsize_multi   # General font size
    plt.rcParams['axes.titlesize'] =16*fontsize_multi   # Title
    plt.rcParams['axes.labelsize'] =14*fontsize_multi   # X/Y label
    plt.rcParams['xtick.labelsize']=12*fontsize_multi   # X ticks
    plt.rcParams['ytick.labelsize']=12*fontsize_multi   # Y ticks
    plt.rcParams['legend.fontsize']=14*fontsize_multi  # Legend
    plt.rcParams['xtick.direction']="in"  # X ticks
    plt.rcParams['ytick.direction']="in"  # X ticks
    plt.rcParams['xtick.top']=True  # X ticks
    plt.rcParams['ytick.right']=True  # X ticks
    #plt.rcParams['lines.linewidth']=3
    #plt.rcParams['lines.markersize']=10
    
    # Size and resolution; layout
    plt.rcParams['figure.figsize'] = [6,4]
    plt.rcParams["axes.formatter.use_mathtext"]=True
    plt.rcParams["figure.constrained_layout.use"]=False
    plt.rcParams["figure.autolayout"]=False
    plt.rcParams["savefig.transparent"] =True
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["figure.dpi"] = 100
    
    # histogram styles
    plt.rcParams["hist.bins"] = 100
    
class Save_fig:
    def __init__(self,fig_prefix=None, exts="pdf,png",dpi=100,SAVE=True):
        self.fig_prefix = fig_prefix
        Path(os.path.dirname(fig_prefix)).mkdir(exist_ok=True)
        self.exts = exts
        self.dpi = dpi
        self.SAVE = SAVE
        pass
    
    def __call__(self,filename):
        """
        Saving figure with various formats.
        """
        self.filename = filename if self.fig_prefix is None else self.fig_prefix+filename
        
        if self.SAVE:
            if len(self.filename.split(".")[-1])==3:
                if self.filename[-3:] not in self.exts:
                    self.exts+=f",{self.filename[-3:]}"
                self.filename=self.filename[:-4]
            for self.ext in self.exts.split(","):
                if self.ext=="pdf":
                    # Disable transparent background of pdf
                    plt.savefig(f"{self.filename}.{self.ext}",dpi=300,transparent=False,bbox_inches='tight')
                else:
                    plt.savefig(f"{self.filename}.{self.ext}",dpi=self.dpi,bbox_inches='tight')
        else:
            return    

def save_fig(filename,exts="pdf,png",dpi=None,SAVE=True):
    """
    Saving figure with various formats.
    """
    #if filename[-4:]==".png":
    #    filename=filename[:-4]
    if SAVE:
        if len(filename.split(".")[-1])==3:
            if filename[-3:] not in exts:
                exts+=f",{filename[-3:]}"
            filename=filename[:-4]
        for ext in exts.split(","):
            if ext=="pdf":
                # Disable transparent background of pdf
                savefig(f"{filename}.{ext}",dpi=300,transparent=False,bbox_inches='tight')
            else:
                savefig(f"{filename}.{ext}",dpi=dpi,bbox_inches='tight')
    else:
        return
    
    
    
def get_font_names():
    return np.unique(sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist]))

def show_all_fonts():
    font_names = get_font_names()
    font_names_counts=len(font_names[:])
    pic_width=12
    pic_height = max(5,font_names_counts//6/2)
    figure(figsize=(pic_width,pic_height))
    # plot(1,1)
    # text(0,0,font,transform=gca().transAxes, fontdict=fontdict)
    warnings.filterwarnings("ignore")
    with warnings.catch_warnings():
        for i,font in enumerate(font_names[:]):
            fontdict = {'fontname':font}
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
            text((i%6)*2/pic_width,(i//6)*0.5/pic_height,font,transform=gca().transAxes, fontdict=fontdict)
        gca().axis('off')
        show()
        
# import matplotlib.font_manager
# from IPython.core.display import HTML

# def make_html(fontname):
#     return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

# # def show_font_html():
# code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])
# HTML("<div style='column-count: 2;'>{}</div>".format(code))


import iminuit
from iminuit import Minuit
from iminuit import *
class Hist:
    def __init__(self):
        pass
    
    def __call__(self,data, xerrs=True,bins=100, *args, **kwargs) :
        """
        Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar
        
        Additional args:
        weight: float
            helper of weights. Unlike weights that needs to be specified for each event,
            you can use a single "weight"
        
        """
        # pop off normed kwarg, since we want to handle it specially
        norm = False
        if 'normed' in kwargs.keys() :
            norm = kwargs.pop('normed')
        # pop off weights kwarg, since we want to handle it specially
        ifweights = False
        if 'weights' in kwargs.keys() :
            ifweights = True
            weights=kwargs["weights"]
            kwargs.pop('weights')  
        if 'weight' in kwargs.keys() :
            ifweights = True
            weights=[kwargs["weight"]]
            kwargs.pop('weight')              

        # retrieve the kwargs for numpy.histogram
        histkwargs = {}
        for key, value in kwargs.items():
            if key in inspect.signature(np.histogram).parameters:
                histkwargs[key] = value
        histkwargs["bins"] = bins

        histvals, binedges = np.histogram(data, **histkwargs )
        yerrs = np.sqrt(histvals)
        yerrs[yerrs==0] = 1

        if norm :
            nevents = float(sum(histvals))
            binwidth = (binedges[1]-binedges[0])
            histvals = histvals/nevents/binwidth
            yerrs = yerrs/nevents/binwidth
        if ifweights :
            histvals = histvals*weights[0]
            yerrs = yerrs*weights[0]     

        bincenters = (binedges[1:]+binedges[:-1])/2

        if xerrs :
            xerrs = (binedges[1]-binedges[0])/2
        else :
            xerrs = None

        # retrieve the kwargs for errorbar
        ebkwargs = {}
        for key, value in kwargs.items() :
            if key in inspect.getfullargspec(plt.errorbar).args or key in ["color","label","markersize"]:
                ebkwargs[key] = value
        out = plt.errorbar(bincenters, histvals, yerrs, xerrs, **ebkwargs,zorder=0)

        if 'log' in kwargs.keys():
            if kwargs['log'] :
                plt.yscale('log')

        if 'range' in kwargs.keys() :
            plt.xlim(*kwargs['range'])
        
        
        self.n=histvals
        self.ibins=binedges
        self.bincenters=bincenters
        self.fig=out
        self.yerrs=yerrs
        return self.n, self.ibins, self.fig
    
    
    def lsq(self,model, range=None, loss='linear'):
        if range is not None:
            mask = (self.bincenters>range[0])&(self.bincenters<range[1])
            self.fit_range=range
        else:
            mask = np.repeat(True,len(self.n))
            self.fit_range=[min(self.bincenters),max(self.bincenters)]
            
        self.model=model
        self.ls = iminuit.cost.LeastSquares(self.bincenters[mask], self.n[mask],self.yerrs[mask], model, loss=loss)
        return self.ls
    
    def fit(self, model, range=None, loss="linear", limits=None, plot=True, **kwargs):
        # retrieve the kwargs for plt.plot
        ebkwargs = {}
        keylist=list(kwargs.keys())
        for key in keylist :
            if key in inspect.getfullargspec(plt.plot).args or key in ["color","label","markersize"]:
                ebkwargs[key] = kwargs[key] 
                kwargs.pop(key)
        
        least_squares = self.lsq(model, range, loss)
        m = Minuit(least_squares,**kwargs)
        if limits is not None:
            for key in limits:
                m.limits[key]=limits[key]
                
        m.migrad()
        self.m=m
        self.func=lambda x: model(x, *m.values)
        
        if plot:
            xplot=np.linspace(self.fit_range[0],self.fit_range[1],200)
            yplot=self.func(xplot)

            plt.plot(xplot,yplot,zorder=1,**ebkwargs)
            
            
# import numpy as np
# import matplotlib.pyplot as plt

# linestyle_str = [
#      ('solid', 'solid'),      # Same as (0, ()) or '-'
#      ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
#      ('dashed', 'dashed'),    # Same as '--'
#      ('dashdot', 'dashdot')]  # Same as '-.'

# linestyle_tuple = [
#      ('densely dotted',        (0, (1, 1))),
#      ('densely dashed',        (0, (5, 1))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
#      ('dotted',        (0, (1, 4))),
#      ('dashed',                (0, (5, 5))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dotted',        (0, (1, 8))),
#      ('long dash with offset', (5, (10, 3))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),]


# def plot_linestyles(ax, linestyles, title):
#     X, Y = np.linspace(0, 100, 10), np.zeros(10)
#     yticklabels = []

#     for i, (name, linestyle) in enumerate(linestyles):
#         ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')
#         yticklabels.append(name)

#     ax.set_title(title)
#     ax.set(ylim=(-0.5, len(linestyles)-0.5),
#            yticks=np.arange(len(linestyles)),
#            yticklabels=yticklabels)
#     ax.tick_params(left=False, bottom=False, labelbottom=False)
#     ax.spines[:].set_visible(False)

#     # For each line style, add a text annotation with a small offset from
#     # the reference point (0 in Axes coords, y tick value in Data coords).
#     for i, (name, linestyle) in enumerate(linestyles):
#         ax.annotate(repr(linestyle),
#                     xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
#                     xytext=(-6, -12), textcoords='offset points',
#                     color="blue", fontsize=8, ha="right", family="monospace")


# fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))

# plot_linestyles(ax0, linestyle_str[::-1], title='Named linestyles')
# plot_linestyles(ax1, linestyle_tuple[::-1], title='Parametrized linestyles')

# plt.tight_layout()
# plt.show()            
        
            
                     
