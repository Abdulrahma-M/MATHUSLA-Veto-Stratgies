import os
import numpy as np
## The Python Standard Library ##
import os,sys
import re
import ast
import copy as cp
from glob import glob
import types
from tqdm import tqdm
from types import ModuleType, FunctionType
from gc import get_referents
from array import array
import warnings
import operator
import inspect
import itertools
import functools
from functools import reduce # import needed for python3; builtin in python2
from collections import defaultdict
from datetime import datetime # Pylab will import datatime directly, so need to put this line after pylab..



import pickle
import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
from scipy.io import loadmat
from scipy import stats,signal
from scipy import optimize
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import uncertainties
import h5py

import ROOT as root
from root_numpy import array2tree

def most_frequent(nums):
    
    """Get the most frequent element in a list"""
    return max(set(nums), key = nums.count) 


def GetFilesInDir(base_dir, postfix=".root"):
    if not (os.path.isdir(base_dir)):
        print("bad directory")
        return []
    files = []
    for file in os.listdir(base_dir):
        if file.endswith(postfix):
            files.append(base_dir + "/" + file)
    return files

def sigfigs(number, n):
    order = int(np.log10(number))
    if order > 0:
        return round(number, n)
    else:
        return round(number, int(-1*order) + n)

def unzip(concatlist, divider=-1):
    lists = []
    n = 0
    for val in concatlist:
        if val == -1:
            n += 1
        else:
            while len(lists) <= n:
                lists.append([])
            lists[n].append(val)
    return lists

def appcut(data, mask):
    data_cut=[]
    for i in range(len(mask)):
        if mask[i]:
            data_cut.append(data[i])
    return data_cut


steel_height=0.03 #m
Box_IP_Depth=85.47#m
def coord_det2sim(vector_xyz):
    """
    input: vector_xyz = (x,y,z), unit is m
    return: transformed (x',y',z'), unit is m
    """
    return np.array([vector_xyz[0]+119.5, vector_xyz[1], -vector_xyz[2]-steel_height])
def coord_sim2cms(vector_xyz):
    """
    input: vector_xyz = (x,y,z), unit is m
    return: transformed (x',y',z'), unit is cm
    """
    return np.array([vector_xyz[1],      -vector_xyz[2]+Box_IP_Depth, vector_xyz[0]])*100 # turn to cm
def coord_det2cms(vector_xyz):
    """
    input: vector_xyz = (x,y,z), unit is m
    return: transformed (x',y',z'), unit is cm
    """    
    return coord_sim2cms(coord_det2sim(vector_xyz))

def coord_sim2det(vector_xyz):
    """
    input: vector_xyz = (x,y,z), unit is m
    return: transformed (x',y',z'), unit is m
    """    
    return np.array([vector_xyz[0]-119.5, vector_xyz[1], -vector_xyz[2]-steel_height])
def coord_cms2sim(vector_xyz):
    """
    input: vector_xyz = (x,y,z), unit is cm
    return: transformed (x',y',z'), unit is m
    """    
    return np.array([vector_xyz[2],      vector_xyz[0], -vector_xyz[1]+Box_IP_Depth*100])/100 # turn to m
def coord_cms2det(vector_xyz):
    """
    input: vector_xyz = (x,y,z), unit is cm
    return: transformed (x',y',z'), unit is m
    """    
    return coord_sim2det(coord_cms2sim(vector_xyz))


# Verification:
# print(coord_det2cms([0,0,1]), coord_cms2det(coord_det2cms([0,0,1])))
# tracks=get_truthtrack(ev)

def theta2eta(theta):
    return -np.log(np.tan(theta/2))


def quadsum(data):
    return np.sqrt(np.sum(np.power(data,2)))

def c2list(cvector):
    return [cvector.at(i) for i in range(cvector.size())]

def sortbyrow(a, irow):
    return a[:,a[irow, :].argsort()]
def sortbycol(a, icol):
    return a[a[:, icol].argsort()]


class Utils:
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
        
        @classmethod
        def purple(cls,ustring):
            return cls.PURPLE + ustring + cls.END
        @classmethod
        def cyan(cls,ustring):
            return cls.CYAN + ustring + cls.END
        @classmethod
        def darkcyan(cls,ustring):
            return cls.DARKCYAN + ustring + cls.END
        @classmethod
        def blue(cls,ustring):
            return cls.BLUE + ustring + cls.END
        @classmethod
        def green(cls,ustring):
            return cls.GREEN + ustring + cls.END
        @classmethod
        def yellow(cls,ustring):
            return cls.YELLOW + ustring + cls.END
        @classmethod
        def red(cls,ustring):
            return cls.RED + ustring + cls.END
        @classmethod
        def bold(cls,ustring):
            return cls.BOLD + ustring + cls.END
        @classmethod
        def underline(cls,ustring):
            return cls.UNDERLINE + ustring + cls.END

    @staticmethod
    def groupby(key,item):
        """
        Group the second list `item` with the first list `key`

        Returns
        -------
        key_grouped
        item_grouped
        """
        data = [(item_i,key_i) for key_i, item_i in zip(key,item)]
        res = defaultdict(list)
        for v, k in data: res[k].append(v)

        key_grouped = [k for k,v in res.items()]
        item_grouped = [v for k,v in res.items()]

        return key_grouped,item_grouped


    @staticmethod
    def is_odd(num):
        return num & 0x1

    @staticmethod
    def getsize(obj):
        """sum size of object & members."""
        # Custom objects know their class.
        # Function objects seem to know way too much, including modules.
        # Exclude modules as well.

        BLACKLIST = type, ModuleType, FunctionType
        if isinstance(obj, BLACKLIST):
            raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
        seen_ids = set()
        size = 0
        objects = [obj]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size

    @staticmethod
    def find_float(string,return_format="float"):
        """
        Find all float numbers inside a string

        return_format: "float" or "str"
        """
        float_in_str = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", string)
        if return_format == "float":
            data = [float(i) for i in float_in_str]
        elif return_format=="str":
            data = [i for i in float_in_str]
        return data

    @staticmethod
    def strike(text):
        return ''.join([u'\u0336{}'.format(c) for c in text])

    @staticmethod
    def getIVserlist(ser0,nser,rawdir='/gpfs/slac/staas/fs1/g/supercdms/tf/nexus/midasdata/NEXUS/R7/Raw/'):
        """
        Get a list of series after an initial series
        """
        fnames = sorted([fname for fname in os.listdir(rawdir) ])
        return fnames[fnames.index(ser0):fnames.index(ser0)+nser]

    @staticmethod
    def get_filelist(series,N_files=None,data_dir="/sdf/group/supercdms/data/CDMS/NEXUS/R8/Raw",verbose=False,data_type=None):
        """
        Get a list of files from the same series

        Inputs
        ------
          series: str
              `series` can be a string of series, or the filename containing wildcards
              * if a series number is give, it will use the default filename pattern for NEXUS combined with `data_dir`
              * if a filename with wildcard is given, it will fetch all files matching the wildcard
          N_files: None, int, tuple or list
              * None: return all filenames of this series
              * int N: return N filenames starting from the first one
              * tuple (N1,N2): return the N1^th to N2^th filenames
              * list: return the filenames following the indices in the list
          data_dir: str
              path to raw data

        Returns
        -------
        file_list: list of string
            a list of filenames
        """
        # List of files
        if "*" in series or (data_type is None):
            file_list = np.array(Utils.sortByExt(glob(series)))
        # case "series_str" is a file... just add in filelist
        elif os.path.isfile(series):
            file_list = [series]
        elif data_type in ["NEXUS"]:
            file_list = np.array(Utils.sortByExt(glob(data_dir+f"/{series}/{series}_F*.mid.gz")))
        elif data_type in ["Animal"]:
            file_list = np.array(Utils.sortByExt(glob(data_dir+f"{series[:8]}/{series}/*.hdf5")))
            #raise Exception("Animal filename pattern not implemented yet. Please make the first arg a filename pattern")
            print("Animal filename pattern not implemented yet. Please make the first arg a filename pattern")

        if N_files is None:
            file_list = file_list[:len(file_list)]
            #n_files = input(f"{len(file_list)} files found. How many files do you want to use?")
            #file_list = file_list[:min(len(file_list),n_files)]
        elif type(N_files) is int:
            if verbose:
                print(f"Using {N_files} out of {len(file_list)} files")
            file_list = file_list[:min(len(file_list),N_files)]
        elif type(N_files) is tuple:
            if N_files[0]>=0:
                if verbose:
                    print(f"Using file {N_files[0]} to {N_files[1]}")
                file_list = file_list[N_files[0]:N_files[1]]
            else:
                if verbose:
                    print(f"Using all files")
        elif type(N_files) is list:
            if verbose:
                print(f"Using file # {N_files}")
            file_list = file_list[N_files]
        return file_list

    @staticmethod
    def sortByExt(files,data_type=None):
        """
        Sort a list of filename by the last number in the filename.

        Inputs
        ------
        files: list of str
            filenames to sort
        data_type: None, "NEXUS" or "Animal"

        Returns
        -------
        filelist:
            sorted filelist
        """
        numbers=list()
        # if data_type=="NEXUS":
        #     for f in files:
        #         numbers.append(int(f.split('_')[-1].split('.')[0][1:]))
        # elif data_type=="Animal":
        #     for f in files:
        #         numbers.append(int(f.split('_')[-1].split('.')[0]))
        # else:
        if files is None or len(files)==0:
            return []
        for f in files:
            numbers.append(float(Utils.find_float(os.path.basename(f).split(".")[0])[-1]))
        return np.array(files)[np.argsort(numbers)].tolist()

    @staticmethod
    def append_dicts(dict1,dict2,verbose=False):
        """
        Append dict2 after dict1
        """
        dict_combined=dict()
        for key in dict1:
            #dict_combined[key]=np.append(dict1[key],dict2[key])
            if key in dict2:
                try:
                    dict_combined[key]=np.concatenate((dict1[key],dict2[key]))#.astype(type(dict2[key][0]))
                except Exception as e:
                    if verbose:
                        print(key,e)
                    continue
        return dict_combined

    @staticmethod
    def flatten1d(a):
        #functools_reduce_iconcat
        """Reduce an array to 1-d"""
        return np.array(functools.reduce(operator.iconcat, a, []))



    @staticmethod
    def rfft_corr(x):
        y=x
        y[1:]*=2
        return y

    @staticmethod
    def downsample_fast(x, q):
        """
        Downsample the signal fast with boxcar filter

        scipy.signal.decimate is the right way for serious downsampling.
        However, when speed is the main consideration it's x10 faster to use
        simple boxcar filter

        Parameters
        ----------
        x : array_like
            The signal to be downsampled, as an 1 dimensional array.
        q : int
            The downsampling factor
        """
        return np.mean(x[:int(q*(len(x)//q))].reshape(-1, q), axis=1)

    @staticmethod
    def kmeans_1d(data,n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters).fit(data.reshape(-1,1))
        kmeans.predict(data.reshape(-1,1))
        clustered = kmeans.cluster_centers_
        return clustered

    @staticmethod
    def center(x):
        return 0.5*(x[1:]+x[:-1])

    @staticmethod
    def roll_fft(a, shift):
        """
        Roll array elements by non-interger number `shift`
        using fft method

        Inputs
        ------
        a: ndarray
        shift: float
            positive is shifting to the right side

        Returns
        -------
        rolled array
        """
        af = np.fft.rfft(a)
        freq=np.fft.rfftfreq(len(a))
        af*= np.exp(-1j*2*np.pi*freq*shift)
        return np.fft.irfft(af)

    @staticmethod
    def roll_zeropad(a, shift, axis=None,pad_value=0):
        """
        Roll array elements along a given axis.

        Elements off the end of the array are treated as pad_value.

        Parameters
        ----------
        a : array_like
            Input array.
        
        shift : int
            The number of places by which elements are shifted.
        
        axis : int, optional
            The axis along which elements are shifted.  By default, the array
            is flattened before shifting, after which the original
            shape is restored.

        Returns
        -------
        res : ndarray
            Output array, with the same shape as `a`.

        See Also
        --------
        roll     : Elements that roll off one end come back on the other.
        rollaxis : Roll the specified axis backwards, until it lies in a
                   given position.

        Examples
        --------
        >>> x = np.arange(10)
        >>> roll_zeropad(x, 2)
        array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
        >>> roll_zeropad(x, -2)
        array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

        >>> x2 = np.reshape(x, (2,5))
        >>> x2
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])
        >>> roll_zeropad(x2, 1)
        array([[0, 0, 1, 2, 3],
               [4, 5, 6, 7, 8]])
        >>> roll_zeropad(x2, -2)
        array([[2, 3, 4, 5, 6],
               [7, 8, 9, 0, 0]])
        >>> roll_zeropad(x2, 1, axis=0)
        array([[0, 0, 0, 0, 0],
               [0, 1, 2, 3, 4]])

        """
        a = np.asanyarray(a)
        if shift == 0: return a
        if axis is None:
            n = a.size
            reshape = True
        else:
            n = a.shape[axis]
            reshape = False
        if np.abs(shift) > n:
            res = np.zeros_like(a)
        elif shift < 0:
            shift += n
            zeros = np.ones_like(a.take(np.arange(n-shift), axis))*pad_value
            res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
        else:
            zeros = np.ones_like(a.take(np.arange(n-shift,n), axis))*pad_value
            res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
        if reshape:
            return res.reshape(a.shape)
        else:
            return res

    @staticmethod
    def get_dt(times1, times2=None, use_which="nearest"):
        """
        calculate times2-times1

        use_which: "before","nearest","after"
        """

        if times2 is None:
            dt_after = np.diff(times1, append=np.inf)
            dt_before = -np.diff(times1, prepend=-np.inf)
        else:
            times2 = np.sort(times2)
            indeces = np.searchsorted(times2, times1)
            dt_after = np.concatenate((times2, [np.inf]))[indeces] - times1
            dt_before = np.concatenate(([-np.inf], times2))[indeces] - times1

        dt_after[np.isinf(dt_after)] = np.inf
        dt_before[np.isinf(dt_before)] = -np.inf

        if np.any(dt_after<0) or np.any(dt_before>0):
            print('In get_dt: ERROR 1. Wrong dt sign')

        if use_which=="nearest":
            use_before = dt_after>-dt_before

            dt = np.zeros_like(dt_after)
            dt[use_before] = dt_before[use_before]
            dt[~use_before] = dt_after[~use_before]
        elif use_which=="before":
            dt = dt_before
        elif use_which=="after":
            dt = dt_after        

        ## revert the meaning of dt here
        return dt
    
    
    @staticmethod
    def merge_intervals(arr, verbose=False):
        """
        https://www.geeksforgeeks.org/merging-intervals/
        # Python3 program to merge overlapping Intervals
        # in O(n Log n) time and O(1) extra space    

        >> Driver code
        >> arr = [[6, 8], [1, 9], [2, 4], [4, 7], [11,12]]
        >> merge_intervals(arr)    
        """

        # Sorting based on the increasing order
        # of the start intervals
        arr.sort(key = lambda x: x[0])

        # array to hold the merged intervals
        m = []
        s = -10000
        max = -100000
        for i in range(len(arr)):
            a = arr[i]
            if a[0] > max:
                if i != 0:
                    m.append([s,max])
                max = a[1]
                s = a[0]
            else:
                if a[1] >= max:
                    max = a[1]

        #'max' value gives the last point of
        # that particular interval
        # 's' gives the starting point of that interval
        # 'm' array contains the list of all merged intervals

        if max != -100000 and [s, max] not in m:
            m.append([s, max])
        if verbose:
            print("The Merged Intervals are :", end = " ")
            for i in range(len(m)):
                print(m[i], end = " ")

        return np.array(m)    
        
    @staticmethod
    def diff_coinc(list1,list2=None,use_nearest=True):
        """
        return closest interval for each events in `list1`
          * if no list2 is provided, the closest one is found in list1
          * if list2 is provided, the closest one is found in list2

        dt =  t1 - t2_closest_to_t1 (positive dt means t1 is after t2)

        Inputs
        ------
        list1: list
            a list of numbers, doesn't neet to be sorted
        
        list2: list
            a list of numbers, doesn't neet to be sorted. If none, calculate dt of list1 alone
        
        use_nearest: boolean
            Only used when there is no list2
            (Default) True: for each event returning the smaller one between the pre- and post- event interval
            Fause: return the post-event interval. The last element is forced to inf

        Returns
        -------
        dts: ndarray, length equal to list1
            If list2 is not provided:
                * closest interval for each event in list1. positive value the means the interval is before the current event.
            If list2 is provided:
                * closest interval for each event in list2. (positive dt means t1 is after t2)
        
        inds2_coinc: list of boolean
            If list2 is not provided:
                the sign of dts
            If list2 is provided:
                the indices of the closest event in list2

        Example
        -------
        >>> diff_coinc([1,2,4,7,10],[1.2,2.4,7,5.5,6,9])
        """
        list1=np.array(list1)
        if list2 is not None:
            list2 = np.array(list2)
            dt=[]
            inds2_coinc=[]
            if len(list1)>0:
                i2_low=0
                inds1=np.argsort(list1);list1=list1[inds1]
                inds2=np.argsort(list2);list2=list2[inds2]
                list2_length=len(list2)
                for trigpt1 in list1:
                    for i2,trigpt2 in enumerate(list2[i2_low:]):
                        if (trigpt1<=list2[i2_low]):
                            # 0. if the first element is already larger than trigpt1
                            dt.append(trigpt1-list2[i2_low])
                            inds2_coinc.append(i2_low)
                            break

                        if i2==(list2_length-i2_low-1):
                            # 1. if already the last one:
                            dt.append(trigpt1-list2[i2_low+i2])
                            break    
                            
                        if (list2[i2_low+i2]<=trigpt1)&(trigpt1<=list2[i2_low+i2+1]):
                            dts = [trigpt1-list2[i2_low+i2],trigpt1-list2[i2_low+i2+1]]
                            #if dts[0]==0:
                            #    dts[0] = trigpt1-list2[i2_low+i2-2]
                            #if dts[1]==0:
                            #    dts[1] = trigpt1-list2[i2_low+i2++1]
                            inds = [i2_low+i2,i2_low+i2+1]
                            dt.append(dts[np.argmin(np.abs(dts))])
                            inds2_coinc.append(inds[np.argmin(np.abs(dts))])
                            break
                    i2_low+=i2
            dt=-np.array(dt)
            inds2_coinc=np.array(inds2_coinc) # reverse the sorting to match the initial list
            inds2_coinc=inds2[inds2_coinc]

        else:
            inds1=np.argsort(list1);times1=list1[inds1]
            dt_after = np.diff(times1, append=np.inf-1)
            dt_before = -np.diff(times1, prepend=-(np.inf-1))
            use_before = dt_after>-dt_before

            dt = np.zeros_like(times1)
            dt[use_before] = dt_before[use_before]
            dt[~use_before] = dt_after[~use_before]
            inds2_coinc=use_before
            dt=-dt if use_nearest else dt_after
        dt=np.array(dt);
        # reverse the sorting to match the initial list
        dt=dt[inds1];
        return dt,inds2_coinc
    

    @staticmethod
    def split(text):
        return list(filter(None,text.split("\n")))

    @staticmethod
    def contrast(a,b,weight=1):
        """return (a-b*weight)/(a+b*weight)
        """
        return (a-b*weight)/(a+b*weight)

    @staticmethod
    def pch(inner,outer,scale=1):
        """return (outer*(1+scale)-inner*(1-scale))/(outer*(1+scale)+inner*(1-scale))
        """
        return (outer*(1+scale)-inner*(1-scale))/(outer*(1+scale)+inner*(1-scale))




    @staticmethod
    def parabola_3points(x1,x2,x3,y1,y2,y3,return_vertex=False):
        """
        Solve for A,B,C for parabola: A x1^2 + B x1 + C = y1 with 3 points
        """
        denom = (x1-x2) * (x1-x3) * (x2-x3);
        A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

        x_vertex = -B / (2*A);
        y_vertex = C - B*B / (4*A);
        if return_vertex:
            return x_vertex,y_vertex

        return A,B,C



    @staticmethod
    def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
        '''
        3-point parabola:
        http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
        '''

        denom = (x1-x2) * (x1-x3) * (x2-x3);
        A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

        return A,B,C

    @staticmethod
    def parabola_interpolation(data,ind):
        """
        parabola interpolation to find the y value at a given x, with 3 points in data around a index (x)

        Inputs
        -------
        data: list of y value
        ind: x value
        """
        if type(ind) is int:
            return data[ind]
        else:
            xa=int(np.floor(ind)); xa=min(xa,len(data)-2)
            xb=int(np.ceil(ind));  xb=min(xb,len(data)-1)
            ya,yb=(data[xa],data[xb])         
            if ind > (xa+xb)/2:
                if xb+1 < len(data):
                    xc = xb+1
                else:
                    xc = xa-1
            else:
                if xa-1 >= 0:
                    xc = xa-1
                else:
                    xc = xb+1
            yc = data[xc]
            A,B,C = Utils.parabola_3points(xa, xb, xc, ya, yb, yc)
            y_vertex = C - B*B / (4*A)
            return y_vertex

    @staticmethod
    def linear_interpolation(data,ind):
        """
        Linear interpolation to find the y value at a given x, with 2 points in data around a index (x)

        Inputs
        -------
        data: list of y value
        ind: x value
        """
        if type(ind) is int:
            return data[ind]
        else:
            xa=int(np.floor(ind)); xa=min(xa,len(data)-2)
            xb=int(np.ceil(ind));  xb=min(xb,len(data)-1)
            ya,yb=(data[xa],data[xb])
            xc=ind
            m = (ya - yb) / (xa - xb)
            yc = (xc - xb) * m + yb
            return yc

    @staticmethod
    def bilinear_interpolation(x, y, points):
        '''Interpolate (x,y) from values associated with four points.

        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.

            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        points = sorted(points)               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0)

    @staticmethod
    def bilinear_interpolation_array(x, y, data):
        if type(x) is int and type(y) is int:
            return(data[x,y])
        x_l,x_h = int(np.floor(x)),int(np.ceil(x))
        y_l,y_h   = int(np.floor(y)),int(np.ceil(y))
        if (x_l==x_h):
            return Utils.linear_interpolation(data[x_l,:],y)
        if (y_l==y_h):
            return Utils.linear_interpolation(data[:,y_l],x)
        return Utils.bilinear_interpolation(x, y, [(x_l,y_l,data[x_l,y_l]),(x_l,y_h,data[x_l,y_h%data.shape[1]]),(x_h,y_l,data[x_h%data.shape[0],y_l]),(x_h,y_h,data[x_h%data.shape[0],y_h%data.shape[1]])])

    @staticmethod
    def lin_interp(x, y, i, half):
        """
        Linear interpolation to find the x value at a given y, with 2 points in data around a index (i) and the y value (half)

        Inputs
        -------
        data: list of y value
        ind: x value
        """
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

    @staticmethod
    def fwhm(x, y, height=0.5):
        """
        Find the interval in x corrsponding to full width half maximum
        x: list of x values
        y: list of y values
        height: default to 0.5 (half maximum)
        """
        half = max(y)*height
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [Utils.lin_interp(x, y, zero_crossings_i[0], half),
                Utils.lin_interp(x, y, zero_crossings_i[1], half)]

    @staticmethod
    def find_crossing(x, y, height=1):
        """
        Find the interval in x corrsponding to a certain value in y
        x: list of x values
        y: list of y values
        height: default to 1
        """
        half = height
        signs = np.sign(np.add(y, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        zero_crossings_i = np.where(zero_crossings)[0]
        return [Utils.lin_interp(x, y, zero_crossings_i[0], half),
                Utils.lin_interp(x, y, zero_crossings_i[1], half)]

    #----------------------------------------------------------------------------
    # A few math functions redefined

    @staticmethod
    def nll_poisson(data, model):
        nll = np.sum([-scipy.stats.poisson.logpmf(k,mu) for k,mu in zip(data,model)])
        return nll

    @staticmethod
    def Uniform(x,A):
        if type(x) not in [np.ndarray,list]:
            return A
        else:
            return A*np.ones_like(x)

    @staticmethod
    def Exp(x,A,t):
        return A*np.exp(-x/t)

    @staticmethod
    def Gauss(x, A, mean, sigma):
        return A * np.exp(-(x - mean)**2 / (2 * sigma**2))

    @staticmethod
    def Gauss_sideband(x, A, mean, sigma, a1,a2):
        # a1 for left, a2 for right
        return Utils.Gauss(x, A, mean, sigma) + sqrt(2*np.pi)*sigma/2*(a1*scipy.special.erfc((x-mean)/sqrt(2)/sigma) + a2*(2-scipy.special.erfc((x-mean)/sqrt(2)/sigma)))

    @staticmethod
    def Poisson(k, Lambda,A):
        # Lambda: mean, A: amplitude
        return A*(Lambda**k/scipy.special.factorial(k)) * np.exp(-Lambda)

    @staticmethod
    def Poly(x, *P):
        '''
        Compute polynomial P(x) where P is a vector of coefficients
        Lowest order coefficient at P[0].  Uses Horner's Method.
        '''
        result = 0
        for coeff in P[::-1]:
            result = x * result + coeff
        return result

    @staticmethod
    def Chi2_reduced(x, dof, A):
        return scipy.stats.chi2.pdf(x*dof,dof)*A

    
    @staticmethod
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
    
    
def pull(x_measure, x_truth, x_unc):
    return (x_measure-x_truth)/x_unc

def poissonerror_div(N1,N2):
    return np.sqrt(1/N1+1/N2)*N1/N2

def chi2_calc(x_est, x_true, err):
    return sum([(x_est[i]-x_true[i])**2/err[i]**2 for i in range(len(err))])


def chi2_list_calc(x_est_list, x_true_list, err):
    chi2=[]
    for i in range(len(x_est_list)):
        chi2.append(((x_est_list[i][0]-x_true_list[i][0])**2+(x_est_list[i][-1]-x_true_list[i][-1])**2)/err[i]**2) 
    return sum(chi2)