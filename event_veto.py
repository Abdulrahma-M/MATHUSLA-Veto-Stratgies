import importlib
from importlib import reload
import os, sys, glob, warnings, glob
import scipy
import numpy as np
import scipy as sp
import joblib
# from tqdm.notebook import tqdm
from tqdm import tqdm
import copy as cp
import iminuit
from iminuit import Minuit
from iminuit import *
import include_modules_root
import ROOT as root

import matplotlib.pyplot as plt
from matplotlib import collections, colors, transforms
import include_modules_root as rt
import track_analysis as TK
from pylab import *
import numpy as np
np.bool=bool
np.float=float
np.object=object
import matplotlib
import visualization, util, event, cutflow, detector
importlib.reload(event)
importlib.reload(visualization)
importlib.reload(util)
importlib.reload(detector)
import scipy as sp




# Gap & Module functions #
gap = 100
module_height = 800
BoxLimits = [  [-5000.0, 5000.0],  [6000.0 + 547, 8917.0 + 547],  [6895.1, 17000.0]]
ModuleYLims=[[6547.0, 6547.0 + module_height], [6547.0 + module_height + gap, 6547.0 + 2*module_height + gap ], [6547.0 + 2*module_height + 2*gap,6547.0 + 3*module_height + 2*gap]] 
ModuleXLims = [ [-4950. + 1000.*n, -4050. + 1000*n] for n in range(10) ]
ModuleZLims = [ [7000.  + 1000.*n,  7900. + 1000*n] for n in range(10) ]
scintillator_height_all = 2.6 # 2cm +0.3*2Al case
wall_gap = 1.0
wall_gap2 = 100.0
wall_height = 2600.0
wall_start_y = ModuleYLims[0][0] - 3
z_min_wall = ModuleZLims[0][0] - wall_gap - scintillator_height_all
cm = 1
LayerYLims= [[6547.0,6547.0+scintillator_height_all],
            [6629.6,6629.6+scintillator_height_all],
            [8547.0,8547.0+scintillator_height_all],
            [8629.6,8629.6+scintillator_height_all],
            [9132.2,9132.2+scintillator_height_all],
            [9214.8,9214.8+scintillator_height_all],
            [9297.4,9297.4+scintillator_height_all],
            [9380.0,9380.0+scintillator_height_all],
            [9462.6,9462.6+scintillator_height_all],
            [9545.2,9545.2+scintillator_height_all]]

floor_midpoint = [(6547.0 + 6547.0+scintillator_height_all)/2,(6629.6 + 6629.6+scintillator_height_all)/2]
wall_midpoint= [(ModuleZLims[0][0]- wall_gap -wall_gap2 - scintillator_height_all*1.5) ,(ModuleZLims[0][0] - wall_gap - scintillator_height_all/2.0)]
xLims=[BoxLimits[0][0],BoxLimits[0][1]]
yLims=[BoxLimits[1][0],BoxLimits[1][1]]
zLims=[BoxLimits[2][0],BoxLimits[2][1]]
def inBox(hit):
    x,y,z=hit[:3]
    if x >= xLims[0] and x <= xLims[1]:
        if y >= yLims[0] and y <= yLims[1]:
            if z >= zLims[0] and z <= zLims[1]:
                return True
        return False


def inGap(hit):
    x,y,z = hit[:3]
    if z < 6998:
        for j in range(len(ModuleYLims)-1):
            if  ModuleYLims[j][1] <= y <= ModuleYLims[j+1][0]:
                return True    
        for k in range(len(ModuleXLims)-1): 
            if  ModuleXLims[k][1] <= x <= ModuleXLims[k+1][0]:
                return True
        return False
    elif y < 6629.6+scintillator_height_all and z >= 6997 :
        for i in range(len(ModuleXLims)-1):
            if  ModuleXLims[i][1] <= x <= ModuleXLims[i+1][0]:
                return True
            if  ModuleZLims[i][1] <= z <= ModuleZLims[i+1][0]:
                return True
        return False
    else:
        return False

    
def inModuleX(xVal):
    for moduleN, moduleLims in enumerate(ModuleXLims):
        if xVal > moduleLims[0] and xVal < moduleLims[1]:
            return moduleN

def inModuleZ(zVal):
    for moduleN, moduleLims in enumerate(ModuleZLims):
        if zVal > moduleLims[0] and zVal < moduleLims[1]:
            return moduleN
        
def inModuleY(yVal):
    for moduleN, moduleLims in enumerate(ModuleYLims):
        if yVal > moduleLims[0] and yVal < moduleLims[1]:
            return moduleN
def inFW(hit):
# returns the layer number given the y value 
# if hit is in the wall returns zero 
# non wall/floor hit inLayer(hit->y) > 2
    yVal= hit[1]
    zVal= hit[2]
    if zVal < 6998:
        return -1

    scintillator_height_all = 2.6
    LayerYLims= [[6547.0,6547.0+scintillator_height_all],
                 [6629.6,6629.6+scintillator_height_all],
                 [8547.0,8547.0+scintillator_height_all],
                 [8629.6,8629.6+scintillator_height_all],
                 [9132.2,9132.2+scintillator_height_all],
                 [9214.8,9214.8+scintillator_height_all],
                 [9297.4,9297.4+scintillator_height_all],
                 [9380.0,9380.0+scintillator_height_all],
                 [9462.6,9462.6+scintillator_height_all],
                 [9545.2,9545.2+scintillator_height_all]]
    for i in range(len(LayerYLims)):
        if yVal > LayerYLims[i][0] and yVal < LayerYLims[i][1]:
            return i+1

    return 0



##  Chi2_ Vertex fumctions ##  


def track_event_sort(data):
    event_pointer=[]
    for ind in data["event_ind"]:
        ind_list=[]
        for i in range(len(data["event_ind"])):
            if data["event_ind"][i]==ind:
                ind_list.append(i)
        event_pointer.append(ind_list)
    return event_pointer


# change it so there is a distance cut rather than a layer cut because of the shared wall index
c=sp.constants.c*1e-7
def init_cov(p2,p1,p2_err,p1_err):
    #(p1_sigma_x,p1_sigma_t,p1_sigma_z,p1_sigma_y,p2_sigma_x,p2_sigma_t,p2_sigma_z,p2_sigma_y)
    
    
    dx=p2[0]-p1[0]
    dy=p2[1]-p1[1]
    dz=p2[2]-p1[2]
    dt=p2[3]-p1[3]
    r=np.sqrt(dx**2+dy**2+dz**2)
    k=r*c
    #dt=r/c
    dt2=pow(dt,2)
    V_diag=[p1_err[0]**2,p1_err[1]**2,p1_err[2]**2,p1_err[3]**2,p2_err[0]**2,p2_err[1]**2,p2_err[2]**2,p2_err[3]**2]
    V=np.diag(V_diag)

    
#    jack=np.array([
#[1      , 0           , 0       , 0       , 0       , 0             , 0     , 0],
#[0       , 1           , 0       , 0       , 0       , 0             , 0     , 0],
#[0       , 0           , 1       , 0       , 0       , 0             , 0     , 0],
#[- 1 / dt, dx / (dt*dt), 0       , 0       , 1 / dt  , - dx / (dt*dt), 0     , 0],
#[0       , dy / (dt*dt), 0       , - 1 / dt, 0       , - dy / (dt*dt), 0     , 1 / dt],
#[0       , dz / (dt*dt), - 1 / dt, 0       , 0       , - dz / (dt*dt), 1 / dt, 0],
#[0       , 0           , 0       , 1       , 0       , 0             , 0     , 0]]) 
    
    
    jac=np.array([[1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [(-dt+(dx*dx/k))/(dt2),    (dx*dy/(k))/(dt2),         (dx*dz/(k))/(dt2),            dx/dt2,       (dt-(dx*dx/k))/(dt2),         -(dx*dy/k)/(dt2),         -(dx*dz/k)/(dt2),     -dx/(dt2)],
                  [(dy*dx/k)/(dt2),          (-dt+(dy*dy/k))/(dt2),     (dy*dz/k)/(dt2),              dy/dt2,      -(dy*dx/k)/(dt2),               (dt-(dy*dy/k))/(dt2),    -(dy*dz/k)/(dt2),     -dy/dt2 ],
                  [(dz*dx/k)/(dt2),          (dz*dy/k)/(dt2),           (-dt+(dz*dz/k))/(dt2),        dz/dt2,      -(dx*dz/k)/dt2,                -(dy*dz/k)/(dt2),          (dt-(dz*dz/k))/dt2,  -dz/dt2 ],
                  [       0,                              0,                              0,                         1,                0,                                   0,                            0,                     0  ]])  
    return (jac.dot(V)).dot(np.transpose(jac))

def chi2_distance2point_err(CovMatrix_track,par,point,point_err):
    #track info par=[x0,y0,z0,t0,vx,vy,vz]
    x_=point[0]
    y_=point[1]
    z_=point[2]
    t_=point[3]
    vx,vy,vz=par[3],par[4],par[5]
    x0,y0,z0,t0=par[0],par[1],par[2],par[6]
    dt=t_-t0
    
    jac=np.array([[1,         0,    0,    dt,  0,   0,   -vx],
                 [0,         1,    0,     0,  dt,  0,    -vy],
                 [0,         0,    1,     0,  0,   dt,   -vz]])    
    
    CovMatrix_vertex = (jac.dot(CovMatrix_track)).dot(np.transpose(jac))
    CovMatrix_vertex[0][0]=CovMatrix_vertex[0][0]+point_err[0]*point_err[0]
    CovMatrix_vertex[1][1]=CovMatrix_vertex[1][1]+point_err[1]*point_err[1]
    CovMatrix_vertex[2][2]=CovMatrix_vertex[2][2]+point_err[2]*point_err[2]
    

    residual_vector= [x_ - (x0+vx*dt), y_- (y0+vy*dt), z_- (z0+vz*dt)]
    error =np.dot(np.transpose(residual_vector),np.dot(np.linalg.pinv(CovMatrix_vertex),residual_vector))
    return error,residual_vector,CovMatrix_vertex


def prop_error(CovMatrix_track,par,point,point_err):
    #track info par=[x0,y0,z0,t0,vx,vy,vz]
    x_=point[0]
    y_=point[1]
    z_=point[2]
    t_=point[3]
    vx,vy,vz=par[3],par[4],par[5]
    x0,y0,z0,t0=par[0],par[1],par[2],par[6]
    dt= t_-t0
    
    jac=np.array([[1,         0,    0,    dt,  0,   0,   -vx],
                 [0,         1,    0,     0,  dt,  0,    -vy],
                 [0,         0,    1,     0,  0,   dt,   -vz]])    
    
    CovMatrix_vertex = (jac.dot(CovMatrix_track)).dot(np.transpose(jac))
    CovMatrix_vertex[0][0]=CovMatrix_vertex[0][0]+point_err[0]*point_err[0]
    CovMatrix_vertex[1][1]=CovMatrix_vertex[1][1]+point_err[1]*point_err[1]
    CovMatrix_vertex[2][2]=CovMatrix_vertex[2][2]+point_err[2]*point_err[2]
    return CovMatrix_vertex
def chi2_error(CovMatrix_track,par,point):
    x_=point[0]
    y_=point[1]
    z_=point[2]
    t_=point[3]
    vx,vy,vz=par[3],par[4],par[5]
    x0,y0,z0,t0=par[0],par[1],par[2],par[6]
    dt= t_-t0
    residual_vector= [x_ - (x0+vx*dt), y_- (y0+vy*dt), z_- (z0+vz*dt)]
    error =np.dot(np.transpose(residual_vector),np.dot(np.linalg.pinv(CovMatrix_track),residual_vector))
    return error
def chi2_distance2point(CovMatrix_track,par,point):
    #track info par=[x0,y0,z0,t0,vx,vy,vz]
    x_=point[0]
    y_=point[1]
    z_=point[2]
    t_=point[3]
    vx,vy,vz=par[3],par[4],par[5]
    x0,y0,z0,t0=par[0],par[1],par[2],par[6]
    dt= t_-t0
    
    jac=np.array([[1,         0,    0,    dt,  0,   0,   -vx],
                 [0,         1,    0,     0,  dt,  0,    -vy],
                 [0,         0,    1,     0,  0,   dt,   -vz]])    
    
    CovMatrix_vertex = (jac.dot(CovMatrix_track)).dot(np.transpose(jac))
    CovMatrix_vertex[0][0]=CovMatrix_vertex[0][0]
    CovMatrix_vertex[1][1]=CovMatrix_vertex[1][1]
    CovMatrix_vertex[2][2]=CovMatrix_vertex[2][2]
    

    residual_vector= [x_ - (x0+vx*dt), y_- (y0+vy*dt), z_- (z0+vz*dt)]
    error = np.transpose(residual_vector).dot(np.linalg.inv(CovMatrix_vertex).dot(residual_vector))
    return error, residual_vector


def uniq_floor(FW):
    j,k=[],[]
    for i in range(len(FW)):
        if not  in_FW(FW[i]) in k:
            j.append(FW[i])
            k.append(in_FW(FW[i]))
        
    if 0 and -1 in k:        
        l1=k.index[-1]
        l2=k.index[0]
        wall=[j[-1],j[0]]
        return wall    
                
    return j
            
        
def time_sort(track):
    time=[]
    sortedd=[]
    for i in range(len(track)):
        time.append(track[i][3])
    for i in range(len(time)): 
        sortedd.append(track[np.argmin(time)])
        time.pop(np.argmin(time))
    return sortedd
        


def LLP(track):
# returns the number of hits a track has in the 6+2 top layers
# if a track has 4 it is either LLP_reconstructable_FW or LLP_reconstructable
    four_count=0
    for hit in track[1]:
        if inLayer(hit) > 2:
            four_count=four_count+1
    if four_count >= 4:
        return True
    else:
        return False


def closest_approach_midpoint(tr1, tr2):

    rel_v = tr2[3:6] - tr1[3:6];
    rel_v2 = np.dot(rel_v, rel_v) 

    displacement = tr2[:3] - tr1[:3]; # position difference
    t_ca = -(  np.dot(displacement, rel_v) - np.dot((tr2[3:6]*tr2[6] - tr1[3:6]*tr1[6]), rel_v)  )/rel_v2;
    
    displacement = tr1[:3] - tr2[:3]; # position difference
    t_ca = (  np.dot(displacement, rel_v) + np.dot((tr2[3:6]*tr2[6] - tr1[3:6]*tr1[6]), rel_v)  )/rel_v2;    
    
    

    pos1 = tr1[:3] + tr1[3:6]*(t_ca - tr1[6]);
    pos2 = tr2[:3] + tr2[3:6]*(t_ca - tr2[6]);

    return (pos1 + pos2)*(0.5),t_ca , np.linalg.norm((pos1- pos2))


def line_dist(tr1,tr2,t_ca):
    pos1 = tr1[:3] + tr1[3:6]*(t_ca - tr1[6]);
    pos2 = tr2[:3] + tr2[3:6]*(t_ca - tr2[6]);  
    displacement = pos1-pos2
    
    return np.dot(displacement,displacement)

cut=cutflow.sample_space("")
def time_config(p2,ip,config):
    hit_layer=cut.in_layer(p2[1])
    ip_track=np.array(np.sqrt((p2[0]-ip[0])**2 + (p2[1]-ip[1])**2 + (p2[2]-ip[2])**2))
    time= p2[3] - (ip_track/c)
    if config ==0:
        dt=p2[3]  
        ip_par=np.array([ip[0],ip[1],ip[2],(p2[0]-ip[0])/dt,(p2[1]-ip[1])/dt,(p2[2]-ip[2])/dt,0])
        err=np.array(detector.Layer().uncertainty(hit_layer))
        ey=2/np.sqrt(12)
        p1=[ip[0],ip[1],ip[2],0]
        p2_err=[err[0]*100,ey,err[2]*100,err[1]]
        p1_err=[0,0,0,0]        
    else:
        dt=(ip_track/c)
        ip_par=np.array([ip[0],ip[1],ip[2],(p2[0]-ip[0])/dt,(p2[1]-ip[1])/dt,(p2[2]-ip[2])/dt,time])
        err=np.array(detector.Layer().uncertainty(hit_layer))
        ey=2/np.sqrt(12)
        p1=[ip[0],ip[1],ip[2],time]
        p2_err=[err[0]*100,ey*100,err[2]*100,err[1]]
        p1_err=[0,0,0,err[1]]
        #p1_err=[0,0,0,err[1]]
        
    return ip_par,p1,p2_err,p1_err


## Event Wise Chi2 Vertex veto ## 


def chi2_vertex(data,event_tracks,cutoff, config="hermetic"):
    rep=[]
    ip=util.coord_sim2cms([0, 0, 85.47])
    if config=="hermetic":
        floor=data["FW"]
    else:
        floor=data["modular"]
    event_chi2=[]
    event_par=[]
    event_err=[]
    for i in event_tracks:
        V_chi2_list=[]
        V_chi2_ip_list=[]
        Vertex_cov_list=[]
        V_chi2_track_list=[]
        par_fit_list=[]
        err_fit_list=[]
        if len(floor[i]) ==0 or data["track_par"][i][7]!=13:
            continue
        for floor_digi in floor[i]:
            p2=floor_digi[:4]
            ip_par,p1,p2_err,p1_err=time_config(p2,ip,1)
            cov=init_cov(p2,p1,p2_err,p1_err)
            track_info=[data["track_cov"][i],np.array(data["track_par"][i])]
            ip_info=[cov,ip_par]
            track_seed=[track_info,ip_info]

        #Minimization step
            def cost(x_,y_,z_,t_):
                error=0
                for track in track_seed:
                    CovMatrix_track=track[0]
                    par=track[1]
                    vx,vy,vz=par[3],par[4],par[5]
                    x0,y0,z0,t0=par[0],par[1],par[2],par[6]
                    dt=t_-t0
                    jac=np.array([[1,         0,    0,    dt,  0,   0,   -vx],
                                 [0,         1,    0,     0,  dt,  0,    -vy],
                                 [0,         0,    1,     0,  0,   dt,   -vz]])    

                    CovMatrix_vertex = (jac.dot(CovMatrix_track)).dot(np.transpose(jac))
                    CovMatrix_vertex[0][0]=CovMatrix_vertex[0][0]+1
                    CovMatrix_vertex[1][1]=CovMatrix_vertex[1][1]+1
                    CovMatrix_vertex[2][2]=CovMatrix_vertex[2][2]+1
                    residual_vector= [x_ - (x0+vx*dt), y_- (y0+vy*dt), z_- (z0+vz*dt)]
                    error = error + np.dot(np.transpose(residual_vector),np.dot(np.linalg.inv(CovMatrix_vertex),residual_vector))
                return error 
            det=detector.Detector()
            mid_point=closest_approach_midpoint(track_info[1],ip_info[1])[0]
            t_ca=closest_approach_midpoint(track_info[1],ip_info[1])[1]

        # make sure the y limit is correct 
            if track_info[1][6] > ip_info[1][6]:
                y_min = ip_info[1][1]
                y_max= ip_info[1][1]
                t_min=ip_info[1][6]
                t_max=track_info[1][6]
            else:
                y_min = track_info[1][1]
                y_max= ip_info[1][1]
                t_max=ip_info[1][6]
                t_min=track_info[1][6]
            m = Minuit(cost,x_= mid_point[0],y_= mid_point[1],z_= mid_point[2],t_=t_ca)
            m.fixed["y_"]=y_min
            m.limits["t_"]=(t_min,t_max)
            m.limits["x_"]=(det.BoxLimits[0][0],det.BoxLimits[0][1])
            m.limits["z_"]=(6895.1,det.BoxLimits[2][1])
            try:
                m.migrad()  # run optimiser
                m.hesse()   # run covariance estimator
                par_fit=np.array(list(m.values))
                err_fit=np.array(list(m.errors))                
                V_chi2_list.append(chi2_distance2point_err(cov,ip_par,par_fit,err_fit)[0] + chi2_distance2point_err(data["track_cov"][i],data["track_par"][i],par_fit,err_fit)[0])
                par_fit_list.append(par_fit)
                err_fit_list.append(err_fit)
            except:
                continue
    # take the FW_digi that minimizes Chi2
        ind=np.argmin(V_chi2_list)
        event_chi2.append(V_chi2_list[ind])

    if len (event_chi2) == 0 :
        return False, event_tracks  
    if min(event_chi2)/2 <= cutoff:
        return True
    return False, event_tracks 



## Functions for event_wise Track projection ## 


def sigma_error(data, event_tracks, cutoff, config = "hermetic"):
    if config == "hermetic":
        floor=data["FW"]
    else:
        floor=data["modular"]
    chi2_list=[]
    for ind in event_tracks:
        if len(floor[ind])==0:
            continue
        central_point=[]    
        track = data["digi_track"][ind][0]
        x,y,z,t =data["digi_track"][ind][0][:4]
        fw=[x[0],y[0],z[0],t[0]]
        if inGap(fw) and config == "modular" :
            continue
        elif inFW(fw)==-1:
            for zval in wall_midpoint:
                central_point.append(TK.wall_projection(zval,track))
        elif  -1 < inFW(fw) <= 2 :
            for yval in wall_midpoint:
                central_point.append(TK.floor_projection(yval,track))               
    # propgate the error to the central point 
        for i in range(len(floor[ind])):
            chi2=[]
            for cental in central_point:
                err=np.array(detector.Layer().uncertainty(cental[1]))
                ey=2/np.sqrt(12)
                err=[err[0]*100,ey,err[2]*100,err[1]]
                p2=cental
                p1=data["track_par"][ind]
                dx=p2[0]-p1[0]
                dy=p2[1]-p1[1]
                dz=p2[2]-p1[2]
                dt=p2[3]-p1[6]
                vx,vy,vz=dx/dt,dy/dt,dz/dt
                par=[p2[0],p2[1],p2[2],vx,vy,vz,p2[3]]
                P_cov=prop_error(data["track_cov"][ind],data["track_par"][ind],cental,err)
                chi2.append(chi2_error(P_cov,par,floor[ind][i]))
        if len(chi2)==0:
            continue
        chi2_list.append(min(chi2))
    if len(chi2_list)==0:
        return False,event_tracks
    event_chi2=min(chi2_list)
    if event_chi2 <= cutoff:
        return True
    return False, event_tracks


## Functions for the event_wise d2g veto  ##

def d2gap(data,event_tracks):
    dist_list=[]
    for ind in event_tracks:
        dist = 99999
        dist_=[]
        central_point=[]    
        track = data["track"][ind][0]
        fw = data["fw_hits"][ind][0][:4]
        fw=[x[0],y[0],z[0],t[0]]
        if inFW(fw)==-1:
            for zval in wall_midpoint:
                central_point.append(TK.wall_projection(zval,track))
        elif  -1 < inFW(fw) <= 2 :
            for yval in floor_midpoint:
                central_point.append(TK.floor_projection(yval,track))
        for cental in (central_point):
            if inBox(cental) != True:
                #out_ofbounds.append(cental)
                continue
            if inGap(cental):
                return True, 0
            elif inFW(fw) ==-1:
                for i in range(2):
                    try:
                        xLim=ModuleXLims[inModuleX(cental[0])]
                        yLim=ModuleYLims[inModuleY(cental[1])]
                    except:
                        continue
                    if abs(cental[0]-xLim[i]) < dist :
                        dist = abs(cental[0]-xLim[i]) 
                    if abs(cental[1]-yLim[i]) < dist :
                        dist = abs(cental[1]-yLim[i])  
                dist_.append(dist)
            elif -1 < inFW(fw) <= 2 :
                for i in range(2):
                    xLim=ModuleXLims[inModuleX(cental[0])]
                    zLim=ModuleZLims[inModuleZ(cental[2])]
                    if abs(cental[0]-xLim[i]) < dist :
                        dist = abs(cental[0]-xLim[i]) 
                    if abs(cental[2]-zLim[i]) < dist :
                        dist = abs(cental[2]-zLim[i])  
                dist_.append(dist)
        if len(dist_)==0:
            continue
        dist_list.append(min(dist_))
    if len(dist)==0:
        return False, event_tracks
    event_dist=min(dist_list)
    return True, event_dist










# first poject the track to the floor 
def truth_chi2_cutoff(data):
    
    fit={}
    fit["digi_track"]=[]
    fit["FW"]=[]
    fit["track_par"]=[]
    fit["track_cov"]=[]
    fit["passed_ind"]=[]
    fit["failed_ind"]=[]
    len_tracks=len(data["track"])
    failed=[]
    passed=[]

    event_chi2=[]
    pointer=track_event_sort(data)
    for indl in tqdm(range(len(pointer))):
        if len(pointer[indl]) ==0:
            continue
        chi2_list=[]   
        for ind in pointer[indl]:
            central_point=[]
            chi2=[]
            track= data["digi_track"][ind][0]
            x,y,z,t = data["digi_track"][ind][0][:4]
            fw=[[x[0],y[0],z[0],t[0]],[x[1],y[1],z[1],t[1]]]
            if inFW(fw[0])==-1:
                for zval in wall_midpoint:
                    central_point.append(TK.wall_projection(zval,track))
            elif  -1 < inFW(fw[0]) <= 2 :
                for yval in wall_midpoint:
                    central_point.append(TK.floor_projection(yval,track))               
    # propgate the error to the central point 
            for cental in central_point:
                for i in range(2):
                    err=np.array(detector.Layer().uncertainty(cental[1]))
                    ey=2/np.sqrt(12)
                    err=[err[0]*100,ey*100,err[2]*100,err[1]]
                    p2=cental
                    p1=data["track_par"][ind]
                    dx=p2[0]-p1[0]
                    dy=p2[1]-p1[1]
                    dz=p2[2]-p1[2]
                    dt=p2[3]-p1[6]
                    vx,vy,vz=dx/dt,dy/dt,dz/dt
                    par=[p2[0],p2[1],p2[2],vx,vy,vz,p2[3]]
                    P_cov=prop_error(data["track_cov"][ind],data["track_par"][ind],cental,err)
                    chi2.append(chi2_error(P_cov,par,fw[i]))    
            if len(chi2) ==0:
                continue
            chi2_list.append(min(chi2))
        if len(chi2_list) ==0:
            continue
        event_chi2.append(min(chi2_list))
    return event_chi2



def sigma_error_list(data, config = "hermetic"):
    
    fit={}
    fit["digi_track"]=[]
    fit["FW"]=[]
    fit["track_par"]=[]
    fit["track_cov"]=[]
    fit["passed_ind"]=[]
    fit["event_ind"]=[]
    fit["failed_ind"]=[]
    fit["track"]=[]
    fit["truth"]=[]
    failed=[]
    passed=[]
    pro=[]
    if config == "hermetic":
        floor=data["FW"]
    else:
        floor=data["modular"]
    event_chi2=[]
    pointer=track_event_sort(data)
    for indl in tqdm(range(len(pointer))):
        if len(pointer[indl]) ==0:
            continue
        chi2_list=[]   
        for ind in pointer[indl]:
            if len(floor[ind])==0:
                continue
            central_point=[]    
            track = data["digi_track"][ind][0]
            x,y,z,t =data["digi_track"][ind][0][:4]
            fw=[x[0],y[0],z[0],t[0]]
            if inGap(fw) and config != "hermetic":
                continue
            elif inFW(fw)==-1:
                for zval in wall_midpoint:
                    central_point.append(TK.wall_projection(zval,track))
            elif  -1 < inFW(fw) <= 2 :
                for yval in wall_midpoint:
                    central_point.append(TK.floor_projection(yval,track))               
        # propgate the error to the central point 
            pro.append(central_point[0])
            for i in range(len(floor[ind])):
                chi2=[]
                for cental in central_point:
                    err=np.array(detector.Layer().uncertainty(cental[1]))
                    ey=2/np.sqrt(12)
                    err=[err[0]*100,ey*100,err[2]*100,err[1]]
                    p2=cental
                    p1=data["track_par"][ind]
                    dx=p2[0]-p1[0]
                    dy=p2[1]-p1[1]
                    dz=p2[2]-p1[2]
                    dt=p2[3]-p1[6]
                    vx,vy,vz=dx/dt,dy/dt,dz/dt
                    par=[p2[0],p2[1],p2[2],vx,vy,vz,p2[3]]
                    P_cov=prop_error(data["track_cov"][ind],data["track_par"][ind],cental,err)
                    chi2.append(chi2_error(P_cov,par,floor[ind][i]))
            if len(chi2)==0:
                continue
            chi2_list.append(min(chi2))
        if len(chi2_list)==0:
            failed.append(pointer[indl])
            continue
        index=np.argmin(chi2_list)
        event_chi2.append(min(chi2_list))
        if chi2_list[index] <= 5:
            passed.append(pointer[indl])
        else:
            failed.append(pointer[indl])
    return len(passed)/len(pointer) , event_chi2,pro