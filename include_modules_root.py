import ROOT
import ROOT as rt
import ROOT as root
from ROOT import TGraphErrors, TCanvas, TLine, TLegend, TF1, gStyle, TH1F
from ROOT import TCanvas, TGraph, TLegend, TGraphErrors, TMath
from ROOT import TF1, TH1F,kGreen, TGraphAsymmErrors
from ROOT import gStyle, gPad, gROOT, TStyle, TPaveStats, TPad
gStyle.SetOptStat(0);
gStyle.SetOptFit(1111);
import root_numpy as npr
import root_pandas as pdr
from root_numpy import array2tree

import types
import numpy as np


def tcanvas(name="", title="", figsize=(800,600)):
    c=TCanvas(name, title ,figsize[0],figsize[1])
    c.Draw()
    return c


def thist(x, bins=None, range=None, name="", title=""):
    
    if type(bins) in [list,np.ndarray]:
        h = TH1F(name, title, len(bins)-1, bins);
    else:
        numbins = min(len(x)//100, 100) if bins is None else bins
        range = [min(x),max(x)] if range is None else range
        h = TH1F(name, title, numbins, range[0], range[1]);
    
    # matplotlib-style wrapper for ROOT functions:
    def xlabel(self,label):
        self.GetXaxis().SetTitle(label)
    h.xlabel = types.MethodType(xlabel,h)
    def ylabel(self,label):
        self.GetYaxis().SetTitle(label)
    h.ylabel = types.MethodType(ylabel,h)
    
    # Add some utilities
    def GetContent(self):
        Nbins=self.GetNbinsX()
        bincenters=np.array([])
        n=np.array([])
        errors=np.array([])
        for i in np.arange(1,Nbins+1):
            i=int(i)
            n=np.append(n,self.GetBinContent(i))
            bincenters=np.append(bincenters,self.GetBinCenter(i))
            errors=np.append(errors,self.GetBinError(i))
        return bincenters,n,errors
    h.GetContent = types.MethodType(GetContent,h)
    
    def SetContent(self,n):
        Nbins=self.GetNbinsX()
        for i in np.arange(1,Nbins+1):
            i=int(i)
            self.SetBinContent(i,n[i-1])
    h.SetContent = types.MethodType(SetContent,h)
    
    def SetError(self,errors):
        Nbins=self.GetNbinsX()
        for i in np.arange(1,Nbins+1):
            i=int(i)
            self.SetBinError(i,errors[i-1])
    h.SetError = types.MethodType(SetError,h)            

            
    
    # Change font size
    h.GetXaxis().SetLabelSize(0.045)
    h.GetXaxis().SetTitleSize(0.045)
    h.GetYaxis().SetLabelSize(0.045)
    h.GetYaxis().SetTitleSize(0.045)
    
    for xx in x:
        h.Fill(xx)
    return h

def tbins(h):
    nbins = h.GetNbinsX()
    n = np.array([h.GetBinContent(i) for i in range(1,nbins+1)])
    ibins = np.array([h.GetBinCenter(i) for i in range(1,nbins+1)])
    return n,ibins




def GetBinCenters(h):
    Nbins=h.GetNbinsX()
    bincenters=np.array([])
    for i in np.arange(1,Nbins+1):
        i=int(i)
        bincenters=np.append(bincenters,h.GetBinCenter(i))  
        
    return bincenters

def GetBinEdges(h):
    Nbins=h.GetNbinsX()
    bin_edges_low=np.array([])
    bin_widths = np.array([])
    for i in np.arange(1,Nbins+1):
        i=int(i)
        bin_edges_low=np.append(bin_edges_low,h.GetBinLowEdge(i))  
        bin_widths=np.append(bin_widths,h.GetBinWidth(i))  
    bin_edges_low=np.append(bin_edges_low,bin_edges_low[-1]+bin_widths[-1])
    return bin_edges_low

def GetBinContents(h):
    Nbins=h.GetNbinsX()
    bincenters=np.array([])
    n=np.array([])
    for i in np.arange(1,Nbins+1):
        i=int(i)
        n=np.append(n,h.GetBinContent(i))        
    return n

def GetBinErrors(h):
    Nbins=h.GetNbinsX()
    bincenters=np.array([])
    errs=np.array([])
    for i in np.arange(1,Nbins+1):
        i=int(i)
        errs=np.append(errs,h.GetBinError(i))        
    return errs

def get_info_c(h):
    """
    Return the bin content, bin centers and bin errors of a histogram
    """
    n,bincenters,errs = GetBinContents(h),GetBinCenters(h),GetBinErrors(h)
    return n,bincenters,errs

def get_info(h):
    """
    Return the bin content, bin edges and bin errors of a histogram
    """    
    n,ibins,errs = GetBinContents(h),GetBinEdges(h),GetBinErrors(h)
    return n,ibins,errs


import ROOT
import array
import root_numpy
def fit_tg(x,y, xerr=None, yerr=None, function="gaus",option="QS", fit_range=None, initial_values=None, bounds=None,
            set_constant=None,
         poisson_yerr=True):
    
    """
    2-D fit using TGraphErrors, support uncertainty on both x and y
    

    INPUT
    ---
    x,y: 
        x and y coordinate
    xerr,yerr:
        uncertainty on x and y. if not specified, xerr=0, yerr=sqrt(y)
    function:
        string. Follow the same convention of ROOT TF1
        
    option:
        string, same as the option of  ROOT::TGraph::Fit. 
        "Q": quiet
        "S":The full result of the fit is returned in the TFitResultPtr. This is needed to get the covariance matrix of the fit. 
        "W":Ignore all point errors when fitting a TGraphErrors or TGraphAsymmErrors
        "R" (Range):Fit using a fitting range specified in the function range with TF1::SetRange.
        "B" (Bound):Use this option when you want to fix one or more parameters and the fitting function is a predefined one (e.g gaus, expo,..), otherwise in case of pre-defined functions, some default initial values and limits are set.
    
    fit_range:
        [fit_start, fit_end]
        
    initial_values:
        [a0, b0, c0,...], initial value of all the parameters in a list
    bounds:
        [[i_par, par1_lower, par1_upper], ...]
    set_constant:
        [[i_par1, par1_value],[i_par2, par2_value], ...]  
        
        
    ---    
    f1 = ROOT.TF1("pol6_2","pol6")
    f1.SetParameters(*initial_values)
    f1.FixParameter(0,0)        
    """
    
    n=len(x)
    if yerr is None and poisson_yerr:
        yerr=np.sqrt(y)
        
    if xerr is not None:
        xerr = np.array(xerr)
        if xerr.ndim==1:
            exl=exh=xerr   
        elif xerr.ndim==2:
            exl,exh=xerr[0], xerr[1]
    else:
        exl=exh=np.zeros(n)
            
    if yerr is not None:
        yerr = np.array(yerr)
        if yerr.ndim==1:
            eyl=eyh=yerr   
        elif xerr.ndim==2:
            eyl,eyh=yerr[0], yerr[1]     
    else:
        eyl=eyh=np.zeros(n)            
    x=array.array("d",x)
    y=array.array("d",y)
    
    g1 = ROOT.TGraphAsymmErrors(n,x,y,exl,exh,eyl,eyh);
    
    # Set initial condition to the function
    func=ROOT.TF1("f1",function)
    if fit_range is not None:
        func=ROOT.TF1("f1",function, *fit_range)
        option = option + "R" if "R" not in option else option
    if set_constant is not None:
        for i_fix in set_constant:
            func.FixParameter(i_fix[0], i_fix[1]);
        option = option + "B" if "B" not in option else option            
    if initial_values is not None:
        func.SetParameters(*initial_values);
        option = option + "B" if "B" not in option else option            
    if bounds is not None:
        for i_bound in bounds:
            func.SetParLimits(*i_bound);
        option = option + "B" if "B" not in option else option            

    fit_res=g1.Fit("f1",option)
    pcov=root_numpy.matrix(fit_res.GetCovarianceMatrix())
    popt = [fit_res.GetParams()[i] for i in range(len(pcov))]
    
    return popt,pcov


def fitu(array, fit_range=None, n_bins=1000, functions=(root.RooGaussian,),
        initial_values=((0.0, 1.0, 1.0),),
        bounds=(((-1e6, 1e6), (0, 1e6), (0, 1e6)), ),
        set_constant=None,
        verbosity=0):
    """Uses the RooFit package to fit a dataset (instead of fitting a histogram)
    Source: Sasha Zaytsev

    Parameters
    ----------
    array : 1-d array or list
        input data array to fit
    fit_range : tuple
        data range for the fit (x_lower, x_upper)
    n_bins : int
        number of points on the x-axis in the output. Does not affect the fit!
    functions : tuple of RooAbsPdf
        Roo pdf function.
        Examples:
        RooGaussian, RooUniform, RooPolynomial, RooExponential
    initial_values : tuple of tuples of floats
        inital values of parameters
        Example:
        functions=(root.RooGaussian, root.RooExponential, root.Polynomial), initial_values=((mean, sigma, a), (exp_k, exp_a), (p1, p2, ..., a))
    bounds : tuple of tuples of tuples of floats
        min and max allowed parameter values
        Example:
        functions=(root.RooGaussian, root.RooExponential), bounds=(((min_mean, max_mean),(min_sig,max_sig),(min_a, max_a)), ((min_k, max_k),(min_a, max_a)))
    set_constant : tuple of tuples of bools   or   None
        whether to fix a certain parameter at a constant value.
        If equals to None, then none of the parameters is fixed
        Example:
        functions=(root.RooGaussian, root.RooExponential), set_constant=((fix_mean, fix_sigma), (fix_k))
    verbosity : int
        verbosity level (might not work. It's tricky)
        -2 - print nothing
        -1 - print errors
         0 - print errors and fit results
         1 - print warnings
         2 - print info

    Returns 
    -------
    x, y, param_values, param_errors
    x : array
        bin centers
    y : array
        fit function values
    param_values : tuple of tuples
        values of fitted parameters. Has the same shape as 'initial_values' arg
    param_values : tuple of tuples
        errors of fitted parameters. Has the same shape as 'initial_values' arg
    """

    # trying to suppress output
    if verbosity < -1:
        root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.FATAL)
    if verbosity == -1 or verbosity == 0:
        root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.ERROR)
    if verbosity == 1:
        root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.WARNING)
    if verbosity >= 2:
        root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.INFO)

    if type(array)==list:
        array = np.array(array)

    if fit_range is None:
        fit_range = (np.min(array), np.max(array))

    # create a tree with one branch
    tree = array2tree(np.array(array, dtype=[('data', np.float64)]))

    data_var = root.RooRealVar('data', 'data', fit_range[0], fit_range[1])
    data_arg_set = root.RooArgSet(data_var)

    dataset = root.RooDataSet('dataset', 'dataset', tree, data_arg_set)

    parameters = []
    roo_functions = []
    amplitudes = []

    # iterating through the functions
    func_names = []
    for i,f in enumerate(functions):
        func_name = f.__name__
        # remove the Roo prefix
        if len(func_name)>3 and func_name[:3]=='Roo':
            func_name = func_name[3:]

        base_func_name = func_name
        k = 2
        while func_name in func_names:
            func_name = '%s%i'%(base_func_name, k)
            k+=1

        func_names.append(func_name)

        # creating function parameters
        func_parameters = []
        for j,initial_value in enumerate(initial_values[i][:-1]):
            name = '%s_p%i'%(func_name, j)
            parameter = root.RooRealVar(name, name, initial_value, *bounds[i][j])
            if not(set_constant is None) and set_constant[i][j]:
                parameter.setConstant(True)
            func_parameters.append(parameter)
        parameters.append(func_parameters)

        # creating function amplitude
        name = '%s_a'%(func_name)
        amplitudes.append(root.RooRealVar(name, name, initial_values[i][-1], *bounds[i][-1]))

        if func_name=='Polynomial':
            roo_functions.append(f(func_name, func_name, data_var, root.RooArgList(*func_parameters)))
        elif func_name=='Uniform' or len(func_parameters)==0:
            roo_functions.append(f(func_name, func_name, data_arg_set))
        else:
            roo_functions.append(f(func_name, func_name, data_var, *func_parameters))

    function_list = root.RooArgList(*roo_functions)
    amplitude_list = root.RooArgList(*amplitudes)
    pdf = root.RooAddPdf('pdf', 'pdf', function_list, amplitude_list)

    # fitting
    fit_results = pdf.fitTo(dataset, root.RooFit.Save(), root.RooFit.Range(*fit_range), root.RooFit.PrintLevel(verbosity-1))
    if fit_results.status()!=0:
        if verbosity>=-1:
            print('----- FIT STATUS != 0 -----')
    if verbosity>=0:
        fit_results.Print()

    tf_parameters = []
    param_values = []
    param_errors = []

    for i,params in enumerate(parameters):
        tf_parameters += params
        param_values.append([p.getVal() for p in params] + [amplitudes[i].getVal()])
        param_errors.append([p.getError() for p in params] + [amplitudes[i].getError()])

    tf_parameters += amplitudes

    tf = pdf.asTF(root.RooArgList(data_var), root.RooArgList(*tf_parameters), data_arg_set)
    a = 0
    for amplitude in amplitudes:
        a += amplitude.getVal()

    bin_w = (fit_range[1] - fit_range[0])/n_bins
    x = np.linspace(fit_range[0]+bin_w/2, fit_range[1]-bin_w/2, n_bins)
    y = np.array([a*tf.Eval(ix) for ix in x])*bin_w

    return x, y, param_values, param_errors  

if 'Fit' in globals():
    Fit.fitu=staticmethod(fitu)
    
    
import ctypes
# from array import array
import copy as cp
def BayesDivide(n_pass, n_total, n_pass_err=None, n_total_err=None):
    """
    Using BayesDivide to calculate the uncertainty of fraction calculation by dividing one number with another.
    
    INPUT:
    ---
    n_pass:
        list of numerator
    n_total:
        list of demonimnator
        
    RETURN:
    ---
    efficiency, efficiency_unc_l, efficiency_unc_h:
        lists of the fraction, the lower sided uncertainty, and the upper sided uncertainty.
    """
    if n_pass_err is None:
        n_pass_err = np.sqrt(n_pass)
        n_total_err = np.sqrt(n_total)
    
    # If the denominator is 0, make it non-zero and set the numerator to 0.
    n_pass = np.array(n_pass)
    n_total = np.array(n_total)
    mask = n_total==0
    n_pass[mask]=0
    n_total[mask]=1
    n_pass_err[mask]=1
    n_total_err[mask]=1
    
    gdiv = ROOT.TGraphAsymmErrors()
    h1 = thist([0], bins=len(n_pass), range=(0,100)) # pass
    h0 = thist([0], bins=len(n_pass), range=(0,100)) # total
    h1.SetContent(n_pass)
    h1.SetError(n_pass_err)
    h0.SetContent(n_total)
    h0.SetError(n_total_err)    

    gdiv.Divide(h1, h0, "c1=0.683 b(1,1) mode");
    # gdiv.Draw()
    # tc.Draw()
    
    # Extract number from TGraph
    n=gdiv.GetN()
    axx=ctypes.c_double(0)
    ayy=ctypes.c_double(0)
    
    efficiency=[]
    efficiency_unc_l=[]
    efficiency_unc_h=[]
    for i in range(0,n):
        gdiv.GetPoint(i,axx,ayy)
        exh=gdiv.GetErrorXhigh(i)
        exl=gdiv.GetErrorXlow(i)
        eyh=gdiv.GetErrorYhigh(i)
        eyl=gdiv.GetErrorYlow(i)
        
        efficiency.append(cp.copy(ayy.value))
        efficiency_unc_l.append(cp.copy(eyl))
        efficiency_unc_h.append(cp.copy(eyh))    
        
    return efficiency, efficiency_unc_l, efficiency_unc_h