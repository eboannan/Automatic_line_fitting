import time
import numpy as np
import numpy.matlib as rep
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy
import pdb
from lmfit import Parameters, minimize

MY_DPI = 300
WAVE_WIDTH = 4
DIFF_ORDER = 2
NUM_OF_PEAKS = 6
LN_2 = np.log(2)
PI = np.pi


def customize_plot():
    plt.close('all')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('font', size=13)
    return


def radial_kernel(x0, X, tau):
    x0 = rep.repmat(x0, X.shape[0], 1)
    return np.exp(-np.sum((X-x0)**2, 1)/(2*tau**2)).reshape(-1, 1)


def local_regression(x0, X, Y, tau):
    x0 = np.concatenate([[1], x0]).reshape(1, -1)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    X = np.column_stack([np.ones((len(X), 1)), X.reshape(-1, 1)])
    rk = np.diag(radial_kernel(x0, X, tau).flatten())
    XW = np.matmul(X.T, rk)
    beta = np.matmul(np.matmul(scipy.linalg.pinv(np.matmul(XW, X)), XW), Y)
    return np.matmul(x0, beta)


def linear_weighted_regression(X, Y, tau, X2, patch_size=20):
    dmn = X2.reshape(-1, 1)
    pred = np.zeros((1, dmn.shape[0]))
    patch = min(X.shape[0], patch_size)
    for i in np.arange(1, dmn.shape[0]+1):
        tindx = np.argsort(np.abs(np.arange(1, X.shape[0])-i))
        patch2 = tindx[range(patch)]
        pred[0, i-1] = local_regression(dmn[i-1], X[patch2], Y[patch2], tau)
    return X, Y, dmn, pred


def diff_umass(x, y, order=1):
    """
    Find the nth derivative of a signal at each point
    Ebo Ewusi-Annan, UMASS Lowell, July 2020
    """
    for n in range(order):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        x = np.row_stack((x[0, 0], x, x[-1, 0]))
        y = np.row_stack((y[0, 0], y, y[-1, 0]))
        interp_x = np.zeros((x.shape[0]-1, 1))
        delta_y = np.zeros((x.shape[0]-2, 1))
        for i in range(interp_x.shape[0]):
            interp_x[i, 0] = x[i, 0] + ((x[i+1, 0] - x[i, 0])/2)
        x[0, 0] = x[0, 0] + (0.00001*x[0, 0])
        x[-1, 0] = x[-1, 0] + (0.00001*x[-1, 0])
        interp_f = scipy.interpolate.interp1d(x.flatten(), y.flatten())
        interp_y = interp_f(interp_x.flatten())
        for j in range(delta_y.shape[0]):
            delta_y[j, 0] = (interp_y[j+1] - interp_y[j])
    return delta_y


def param_transform(param):
    """
    transform paramters  for lmfit
    """
    pa = np.zeros((1, len(param)))
    for i in range(len(param)):
        pa[0, i] = param[F"p{i}"]


def funeval(param, lam, amplitude_guess, center_guess,
            num_params=4, mask=None):
    """
    Voigt profile using real part of fadeeva function
    """
    if param.ndim == 1:
        param = param.reshape(-1, 1)
    if amplitude_guess.ndim == 1:
        amplitude_guess = amplitude_guess.reshape(-1, 1)
    if center_guess.ndim == 1:
        center_guess = center_guess.reshape(-1, 1)
    if param.shape[0] == 1:
        param = param.reshape(int(param.shape[1]/num_params), -1).T
    v_profile = np.zeros((len(lam), param.shape[1]))
    for i in range(param.shape[1]):
        try:
            lam_o = center_guess[i, 0] - param[0, i]
        except Exception:
            pdb.set_trace()
        amplitude = amplitude_guess[i, 0]*param[1, i]
        gaussian_width = param[2, i]
        lorentzian_width = param[3, i]
        lam_corrected = lam - lam_o

        if num_params == 5:
            offset = param[4, i]
        if num_params == 6:
            slope = param[5, i]
            offset = param[4, i]
        x = lam_corrected*(2*np.sqrt(LN_2))/gaussian_width
        y = (lorentzian_width/gaussian_width)*(np.sqrt(LN_2))
        z = x+y*1j
        real_fadeeva = scipy.special.wofz(z).real
        voigt = (2*np.sqrt(LN_2/PI))*(real_fadeeva/gaussian_width)
        normalized_voigt = voigt/np.max(voigt)
        if num_params == 5:
            voigt = amplitude*normalized_voigt + offset
        elif num_params == 6:
            voigt = amplitude*normalized_voigt + slope*lam + offset
        else:
            voigt = amplitude*normalized_voigt
        if mask is not None:
            pass  # not implemented yet
        v_profile[:, i] = voigt
    return np.sum(v_profile, axis=1)


def objective_function(param, lam, y_raw, amplitude_guess,
                       center_guess, differentiate, num_params=4, peak_weight=10,
                       mu=0.6, threshold=0.01):
    param = param_transform(param)
    y_eval = y_evaluation(param, lam, amplitude_guess, center_guess,
                          False, num_params=4, mask=None)
    if differentiate:
        y_eval = diff_umass(lam, y_eval, order=2)
        chi = np.sqrt(np.abs(y_raw-y_eval))
        for h in range(len(center_guess)):
            b = np.argmax(lam == center_guess[h])
            chi[b] = chi[b]*peak_weight
    else:
        delta_y_raw = diff_umass(lam, y_raw[:, 0], order=2)
        delta_y_new = diff_umass(lam, y_eval, order=2)
        chi_1 = (1-mu)*(np.sqrt(np.abs(delta_y_raw-delta_y_new)))
        chi_2 = mu*np.sqrt(np.abs(y_raw[:, 1]-y_eval))
        for u in range(len(chi_2)):
            if abs(chi_2[u]) > threshold:
                chi_2[u] = chi_2[u]*peak_weight
        chi = chi_1+chi_2
    return np.sqrt(chi)


def y_evaluation(param, lam, amplitude_guess, center_guess,
                 differentiate, num_params=4, mask=None):
    y_eval = funeval(param, lam, amplitude_guess, center_guess,
                     num_params=num_params)
    if differentiate:
        return diff_umass(lam, y_eval, order=2)
    else:
        return y_eval


def vary_or_fix(i, numpar=None, tofix=None, pk=True):
    # Generate boolean for lmfit whether to fix or vary a parameter
    if pk is False:
        return False
    else:
        if tofix is None:
            return True
        else:
            if numpar is None:
                return not(i in np.array(tofix))
            else:
                return not(((i-numpar) % numpar) in np.array(tofix))


def get_fitted_params(results, num_params):
    # retrieve fitted parameters from lmfit results
    params = np.zeros(num_params)
    for i in range(num_params):
        params[i] = results.params[F"p{i}"].value
    return params


def range_for_fit(lam, y, peak_value, near_pks, wav=5):
    # limit range of data used for fitting to speed the fitting process
    minW = np.max(np.array([np.min(near_pks)-(wav), np.min(lam)]))
    maxW = np.min(np.array([np.max(near_pks)+(wav), np.max(lam)]))
    lower_lim = lam >= minW
    upper_lim = lam <= maxW
    return lam[lower_lim and upper_lim], y[lower_lim and upper_lim]


def my_plot(lam, data=None, num=None, fig_size=(4, 3), grid_spec=None,
            height_ratios=None, labels=None, xlabel=None, y_label=None):
    plt.figure(num=num, figsize=fig_size)
    gs = gridspec.GridSpec(grid_spec[0], grid_spec[1], height_ratios=height_ratios)
    if labels is None:
        labels = []
        for i in range(len(data)):
            labels.append(F"label_{i}")
    for i in range(len(data)):
        plt.subplot(gs[i])
        plt.plot(lam,data[i],'r', lw=1.5, label='actual')
        plt.plot(lam,data_2,'b', lw=1.5, label='fit')
 
    if differentiate:
        plt.ylim(-0.3,0.3)
        plt.ylabel('Intensity deriv.') 
    else:
        plt.ylim(0.0,1.1)
        plt.ylabel('Norm. intensity') 
    plt.xlim(min(lam),max(lam))
    plt.ylim(0.0,1.1)
    plt.ylabel('Norm. intensity') 
    plt.subplot(gs[1])      
    plt.plot(lam,(data-diffynew),'k-',lw=1.5,label='residual')
    plt.xlabel('Wavelength (nm)')
    if differentiate:
        plt.ylim(-0.5,0.5)
    else:
        plt.ylim(-0.5,0.5)
    plt.xlim(min(lam),max(lam))
    plt.ylabel('res.')
    plt.tight_layout()

def my_plot(lam, data_1, data_2, differentiate, fig_size=(4, 3)):
    if differentiate:
        plt.figure(num=4, figsize=fig_size)
    else:
        plt.figure(num=44, figsize=fig_size)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    plt.subplot(gs[0])
    plt.plot(lam,data_1,'r', lw=1.5, label='actual')
    plt.plot(lam,data_2,'b', lw=1.5, label='fit')
 
    if differentiate:
        plt.ylim(-0.3,0.3)
        plt.ylabel('Intensity deriv.') 
    else:
        plt.ylim(0.0,1.1)
        plt.ylabel('Norm. intensity') 
    plt.xlim(min(lam),max(lam))
    plt.ylim(0.0,1.1)
    plt.ylabel('Norm. intensity') 
    plt.subplot(gs[1])      
    plt.plot(lam,(data-diffynew),'k-',lw=1.5,label='residual')
    plt.xlabel('Wavelength (nm)')
    if differentiate:
        plt.ylim(-0.5,0.5)
    else:
        plt.ylim(-0.5,0.5)
    plt.xlim(min(lam),max(lam))
    plt.ylabel('res.')
    plt.tight_layout()

def EF(data1,tau,intgName,peak,count,pkfall,differentiate,prior,minWave,maxWave,max_data,num_params =4):

    customize_plot()
    

    
    
    def my_plot_diff():
        from matplotlib import gridspec
        plt.figure(num=45, figsize=(4,3))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        plt.subplot(gs[0])
        diff_data = diff_umass(lam,data,order=2)
        diff_diffynew = diff_umass(lam,diffynew,order=2)
        plt.plot(lam,diff_data ,'r',lw=1.5, label='actual')
        plt.plot(lam,diff_diffynew,'b',lw=1.5,label='fit')
        plt.ylim(-0.4,0.25)
        plt.ylabel('Intensity deriv.')      
        plt.xlim(min(lam),max(lam))
        if max(lam)<400:
            plt.xlim(258,264)
        if max(lam)>400 and max(lam)<900:
            plt.xlim(392,398)
        if  max(lam)>700:
            plt.xlim(760,785)
        plt.xlabel('Wavelength (nm)')
        
    def my_plot2(differentiate):
        if differentiate:
            plt.figure(num=5, figsize=(4,3))
        else:
            plt.figure(num=55, figsize=(4,3))
        plt.ylabel(r'Intensity $\times 10^\mathrm{12}$ (arb. units)') 
        plt.xlabel('Wavelength (nm)')
        plt.ylim(0,max((data*max_data)/1e12))
        plt.tight_layout()
    
    if differentiate:
        Name1=intgName['raw_smooth']
        Name2=intgName['peaks']
        Name3=intgName['diff_fit']
        Name4=intgName['diff_fit_peaks']
        Name5=intgName['diff_spec']
    else:
        Name3=intgName['fit']
        Name4=intgName['fit_peaks']
    lam = data1[:,0]
    data = data1[:,1]
    _,_,_,pred=linear_weighted_regression(lam,data,tau,lam)
    count +=1
 
    if (peak == 'manual'): 
        if count > 1:
            pksx = pkfall['x']
            pksy = pkfall['y']
        else:
            plt.figure(num=2,figsize=(10,15))
            plt.plot(lam.reshape(-1,1),pred.reshape(-1,1),'g',lw=0.8)
            pks=plt.ginput(n=-1,timeout=0)
            plt.show()
            pksx =np.zeros((len(pks),1))
            pksy =np.zeros((len(pks),1))
            for u in range(len(pks)):
                pksx[u,0] = pks[u][0]
                pksy[u,0] = pks[u][1]
    else:
        if (peak == 'autorep') and (count > 1):
            pksx = pkfall['x']
            pksy = pkfall['y']
        else:
            ycwt = pred.reshape(-1,1).flatten()
            pksin,_ = scipy.special.find_peaks(ycwt, prominence=0.0)
            pksx = lam[pksin].reshape(-1,1)
            pksy = ycwt[pksin].reshape(-1,1)
        
    sorting = np.argsort(pksx.flatten())# sorting uneccesary for automatic but may be for manual peak selection
    pksx = pksx[sorting] 
    pksy = pksy[sorting]
    pred = pred.reshape(-1,1)
    amplitude_guess = pksy #Amplitude constant
    center_guess = pksx #Wavelength constant
    if differentiate:
        plt.figure(num=1,figsize=(4,3), dpi=MY_DPI)
        plt.plot(lam,data,'r',lw=1.5, label='spectrum')
        plt.plot(pksx,pksy,'k.',markersize=5, label='peak')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r' Norm. intensity')
        plt.tight_layout()
        plt.xlim(min(lam),max(lam))
        plt.ylim(0,max(data))
        plt.show()
        plt.savefig(Name1+'.svg', format='svg')
        plt.savefig(Name1+'.tiff',dpi=MY_DPI)
        
        
        plt.figure(num=2,figsize=(4,3))       
        plt.plot(lam,(data*max_data)/1e12,'r',lw=1.5, label='raw')
        plt.plot(lam,(pred*max_data)/1e12,'b',lw=1.5,label='smooth')     
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r'Intensity $\times 10^\mathrm{12}$ (arb. units)')  
        plt.tight_layout()
        plt.xlim(min(lam),max(lam))
        plt.ylim(0,max((data*max_data)/1e12))
        plt.show()
        plt.savefig(Name2+'.svg', format='svg', dpi=MY_DPI)
        plt.savefig(Name2+'.tiff',dpi=MY_DPI)

    data_t =np.zeros((len(lam),2))
    data_t[:,0] = lamV
    data_t[:,1] = pred.flatten()
    if differentiate is True:
        raw_data = np.column_stack((data_t[:,0],diff_umaspatch_sizes(data_t[:,0], data_t[:,1],order=2))) 
    else:
        raw_data = data_t
    pks_sortVal =np.ones((num_params,len(pksx)),dtype=int)
    lam_mask =np.zeros((len(lam),len(pksx))) 
    if len(prior['paramVal']) == 0:
       paramVal =np.zeros((num_params,len(pksx))) 
       paramVal[0,:] = 0.001
       paramVal[1,:] = 0.5
       paramVal[2,:] = 0.1
       paramVal[3,:] = 0.1
       if num_params==5:
           paramVal[4,:] = 0.01
       if num_params==6:
           paramVal[5,:] = 0.01
       
       lbVal =np.zeros((num_params,len(pksx))) 
       lbVal[0,:] = -0.5
       if num_params==6:
           lbVal[5,:] = -10.0

       
       ubVal =np.zeros((num_params,len(pksx))) 
       ubVal[0,:] = 0.5
       ubVal[1,:] = 1.0
       ubVal[2,:] = 1.0
       ubVal[3,:] = 5.0
       if num_params==5:
           ubVal[4,:] = 1.0
       if num_params==6:
           ubVal[4,:] = 1.0
           ubVal[5,:] = 10.0
    else:
        paramVal = prior['paramVal']
        lbVal = prior['lbVal']
        ubVal = prior['ubVal']
    start = time.time()
    for k in range(len(pksx)):  
        if differentiate is True:   
            param=       paramVal[:,max(0,k-(DIFF_ORDER-1)):min(len(pksy),k+DIFF_ORDER)]#   % Positions
            pks_sort= pks_sortVal[:,max(0,k-(DIFF_ORDER-1)):min(len(pksy),k+DIFF_ORDER)]#   % Positions
            lb =            lbVal[:,max(0,k-(DIFF_ORDER-1)):min(len(pksy),k+DIFF_ORDER)]
            ub =            ubVal[:,max(0,k-(DIFF_ORDER-1)):min(len(pksy),k+DIFF_ORDER)]
        else:
            param=       paramVal[:,max(0,k-(NUM_OF_PEAKS-1)):min(len(pksy),k+NUM_OF_PEAKS)]#   % Positions
            pks_sort= pks_sortVal[:,max(0,k-(NUM_OF_PEAKS-1)):min(len(pksy),k+NUM_OF_PEAKS)]#   % Positions
            lb =            lbVal[:,max(0,k-(NUM_OF_PEAKS-1)):min(len(pksy),k+NUM_OF_PEAKS)]
            ub =            ubVal[:,max(0,k-(NUM_OF_PEAKS-1)):min(len(pksy),k+NUM_OF_PEAKS)]

        y = raw_data[:,1]
        fit_params = Parameters()
        param = param.T.reshape(-1,1).flatten()
        pks_sort = pks_sort.T.reshape(-1,1).flatten()
        ub = ub.T.reshape(-1,1).flatten()
        lb = lb.T.reshape(-1,1).flatten()
        
        if differentiate is True:
            mnk = np.max([0,k-(DIFF_ORDER-1)])
            msk = np.min([len(pksy),k+DIFF_ORDER])
            for i in range(len(param)):
                fit_params.add('p'+str(i), value=param[i], max=ub[i], min=lb[i], vary=vary_or_fix(i,numpar =num_params,tofix=None,pk=bool(pks_sort[i])))
           
            trunc_lam,trunc_y = range_for_fit(lam,y,pksx[k],center_guess[mnk:msk],wav=WAVE_WIDTH)
            p_min = minimize(objective_function,fit_params,method='least_squares',args=(trunc_lam,trunc_y,amplitude_guess[mnk:msk],center_guess[mnk:msk],differentiate,num_params,))
            p_min = get_fitted_params(p_min,len(param)).reshape((num_params,-1),order='F')
            if (p_min.shape[1]<((DIFF_ORDER*2)-1) and k<(DIFF_ORDER-1)):
                paramVal[:,k]= p_min[:,k]
            else:
                paramVal[:,k]= p_min[:,(DIFF_ORDER-1)]                 
            pks_sortVal[:,k]= 0          
        else:
            mnk = np.max([0,k-(NUM_OF_PEAKS-1)])
            msk = np.min([len(pksy),k+NUM_OF_PEAKS])
            for i in range(len(param)):
                # pdb.set_trace()
                fit_params.add('p'+str(i), value=param[i], max=ub[i], min=lb[i], vary=vary_or_fix(i,numpar=num_params,tofix=None,pk=bool(pks_sort[i])))
            trunc_lam,trunc_y = range_for_fit(lam,y,pksx[k],center_guess[mnk:msk],wav=WAVE_WIDTH)
            _,trunc_d = range_for_fit(lam,data,pksx[k],center_guess[mnk:msk],wav=WAVE_WIDTH)
            new_y = np.hstack((trunc_y.reshape(-1,1),trunc_d.reshape(-1,1)))

            p_min = minimize(objective_function,fit_params,method='least_squares',args=(trunc_lam,new_y,amplitude_guess[mnk:msk],center_guess[mnk:msk],differentiate,num_params,))
            p_min = get_fitted_params(p_min,len(param)).reshape((num_params,-1),order='F')
            if (p_min.shape[1]<((NUM_OF_PEAKS*2)-1)  and k<(NUM_OF_PEAKS-1)):
                try:
                    paramVal[:,k]= p_min[:,k]
                except IndexError:
                    pdb.set_trace()
            else:
                paramVal[:,k]= p_min[:,(NUM_OF_PEAKS-1)]      
            pks_sortVal[:,k]= 0
        mg = np.zeros_like(lam).copy()
        magic_lam=lam.searchsorted(trunc_lam) # wavelength selector for wavelength used in fitting
        mg[magic_lam]=1.0
        lam_mask[:,k] = mg

   
    diffynew = y_evaluation(paramVal,lam,amplitude_guess,center_guess,False,num_params=num_params,mask=lam_mask)
    ynew2  = funeval(paramVal,lam,amplitude_guess,center_guess,num_params=num_params, mask=lam_mask)
    lam4 =np.arange(np.min(lam),np.max(lam),0.001)
    ynew4 = funeval(paramVal,lam4,amplitude_guess,center_guess,num_params=num_params,mask=lam_mask)
    stop = time.time()
    print(f'time taken:{(stop-start)} s')


    my_plot(differentiate)
    plt.savefig(Name3+'.tiff',dpi=MY_DPI)
    plt.savefig(Name3+'.svg', format='svg', dpi=MY_DPI)

    my_plot2(differentiate)
    integ = np.zeros((paramVal.shape[1],5))
    for t in range(paramVal.shape[1]):
        p_min  = paramVal[:,t]
        fwhm= 0.5346*p_min[3]+np.sqrt(0.2166*p_min[3]**2+p_min[2]**2)# Voigt FWHM
        p_min2 = center_guess[t]-p_min[0]
        lam2 = np.arange(p_min2-20*fwhm,p_min2+20*fwhm,0.01).flatten()
        ynew3 = funeval(p_min,lam2,amplitude_guess[t],center_guess[t],num_params=num_params)
        integ[t,0] =p_min2
        integ[t,1] = p_min[1]*amplitude_guess[t]
        integ[t,2]= p_min[2]
        integ[t,3] = p_min[3]
        integ[t,4] =np.trapz(lam2)*max_data
        plt.plot(lam2,(ynew3*max_data)/1e12,color='green',linestyle='--',lw=1)
    plt.plot(lam2,(ynew3*max_data)/1e12,color='green',linestyle='--',lw=1, label='profile') # hact to give the individua peaks label    
    plt.plot(lam,(data*max_data)/1e12,'r-',lw=1.5, label='actual')
    plt.plot(lam4,(ynew4*max_data)/1e12,'b-',lw=1.5,label='sum of profiles')
    plt.xlim(min(lam),max(lam))
    plt.tight_layout()
    plt.savefig(Name4+'.svg', format='svg', dpi=MY_DPI)
    plt.savefig(Name4+'.tiff',dpi=MY_DPI)
    if max(lam)<400:
        plt.xlim(258,264)
        plt.ylim(0,4.5)
    if max(lam)>400 and max(lam)<900:
        plt.xlim(392,398)
    if  max(lam)>700:
        plt.xlim(760,785)

    plt.savefig(Name4+'zoom.svg', format='svg', dpi=MY_DPI)
    plt.savefig(Name4+'zoom.tiff',dpi=MY_DPI)
    if differentiate:
        my_plot_diff()
        plt.savefig(Name5+'zoom.svg', format='svg', dpi=MY_DPI)
        plt.savefig(Name5+'zoom.tiff',dpi=MY_DPI)
    output={}
    output['fit']=np.hstack((lam.reshape(-1,1),(ynew2*max_data).reshape(-1,1)))
    output['actual']=np.hstack((lam.reshape(-1,1),(data*max_data).reshape(-1,1)))
    output['pksx'] = pksx
    output['pksy'] = pksy
    #Create new bounds for next fitting if fitting similar spectra
    ubVal[1,:]  = 100*paramVal[1,:]
    ubVal[2,:]  =100*paramVal[2,:]
    ubVal[3,:]  = 100*paramVal[3,:] 
    output['lbVal'] = lbVal
    output['ubVal'] = ubVal
    output['paramVal'] = paramVal
    output['Integ'] = integ
    output['count'] = count
    paramVal=[]
    integ=[]
    lbVal=[]
    ubVal=[]
    pksx =[]
    pksy =[]
    return output
    
    

        
        
        
    
    
    
