def EF(data1,tau,intgName,peak,count,pkfall,Diff_space,prior,minWave,maxWave,max_data,numparam =4):
    import numpy as np
    import matplotlib
    import numpy.matlib as rep
    import matplotlib.pyplot as plt
    from scipy import linalg, interpolate
    from scipy.signal import find_peaks#, find_peaks_cwt
    import matplotlib.ticker as mticker
    from scipy import special
    import time
    import pdb
    from lmfit import Parameters, minimize
    my_dpi = 300
    plt.close('all')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('font', size=13)          # controls default text sizes

    trunc_wav =4 # used to limit the wavelength range considered during fitting
    diff_peaks = 2 # order of differentiation
    raw_peaks =6 # number of peaks to consider
    def radial_kernel(x0,X,tau):
        x0 = rep.repmat(x0,X.shape[0],1)
        return np.exp(-np.sum((X-x0)**2,1)/(2*tau**2)).reshape(-1,1)
    
    def local_regression(x0,X,Y,tau):
        x0 = np.concatenate([[1],x0]).reshape(1,-1)
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        X = np.column_stack([np.ones((len(X),1)),X.reshape(-1,1)])
        #we =radial_kernel(x0,X,tau)
        rk = np.diag(radial_kernel(x0,X,tau).flatten())
        XW = np.matmul(X.T,rk)
        beta = np.matmul(np.matmul(linalg.pinv(np.matmul(XW,X)),XW),Y)
        return np.matmul(x0,beta)

    def LinearWeightedRegression(X,Y,tau,X2):  
        # do linear weighted regression
        dmn = X2.reshape(-1,1)
        pred =np.zeros((1,dmn.shape[0]))
        patch = min(X.shape[0],20)
        for i in np.arange(1,dmn.shape[0]+1):
            tindx =np.argsort(np.abs(np.arange(1,X.shape[0])-i))
            patch2 =tindx[range(patch)]
            pred[0,i-1]= local_regression(dmn[i-1],X[patch2],Y[patch2],tau)          
        return X,Y,dmn,pred
    
    def diff_umass(x,y,order=1):
        #Find the nth derivative of a signal at each point
        # Ebo Ewusi-Annan, UMASS Lowell, July 2020
        for n in range(order):
            x_or = x.reshape(-1,1)
            x =x.reshape(-1,1)
            y=y.reshape(-1,1)
            x= np.row_stack((x[0,0],x,x[-1,0]))
            y= np.row_stack((y[0,0],y,y[-1,0]))
            interp_x = np.zeros((x.shape[0]-1,1))
            diff_y = np.zeros((x.shape[0]-2,1))
            
            for i in range(interp_x.shape[0]):
                interp_x[i,0]= x[i,0]+ ((x[i+1,0]-x[i,0])/2)

            x[0,0] = x[0,0]+(0.00001*x[0,0]) # nudge it a bit to enable interpolation
            x[-1,0] = x[-1,0]+(0.00001*x[-1,0])
            interp_f = interpolate.interp1d(x.flatten(),y.flatten())
            interp_y= interp_f(interp_x.flatten())
            for j in range(diff_y.shape[0]):
                diff_y[j,0] = (interp_y[j+1]-interp_y[j])
            x =x_or
            y = diff_y
        return diff_y
    
    def param_transform(param):
        # transform paramters  for lmfit
        pa = np.zeros((1,len(param)))
        for i in range(len(param)):
            pa[0,i]=param['p'+str(i)]
        return pa
    
    def funeval(param,lam,AmpConst,WveConst,pseudo_Voigt =False,num_param=4, mask=None):
        # Voigt profile using real part of fadeeva function
        if param.ndim==1:  # handl1 1 D shapes
            param=param.reshape(-1,1)
          
        if AmpConst.ndim==1:
            AmpConst = AmpConst.reshape(-1,1)
            
        if WveConst.ndim==1:
            WveConst = WveConst.reshape(-1,1)
            
        if param.shape[0]==1:
            param = param.reshape(int(param.shape[1]/num_param),-1).T
        v_profile = np.zeros((len(lam),param.shape[1]))
        for i in range(param.shape[1]):
            try:
                lam0 = WveConst[i,0]-param[0,i] #peak position
            except:
                pdb.set_trace()
            s= AmpConst[i,0]*param[1,i]     #intensity
            Lg = param[2,i]                 #Gaussian width
            Ll = param[3,i]                 #Lorentzian width
            aD = Lg
            vv0 =lam- lam0
            ln2 = np.log(2)
            pi = np.pi
            if num_param==5:
                offset =param[4,i] 
            if num_param ==6:
                slope =param[5,i] 
                offset =param[4,i] 
            if pseudo_Voigt:
                x = vv0
                L = (Lg**5+2.69269*(Lg**4)*Ll+2.42843*(Lg**3)*(Ll**2)+
                         4.47163*(Lg**2)*(Ll**3)+0.07842*Lg*(Lg**4)+Ll**5)**(1/5)

                rho = 1.36603*(Ll/L)-0.47719*(Ll/L)**2 + 0.11116*(Ll/L)**3
                fG = (1/(np.sqrt(pi)*Lg))*np.exp(-x**2/Lg**2)
                fL = (1/(pi*Ll))/(1+(x**2/Ll**2))
                v_profile[:,i]= (1-rho)*fG + rho*fL
            else:   
                x= vv0*(2*np.sqrt(ln2))/aD
                y=(Ll/Lg)*(np.sqrt(ln2))
                z= x+y*1j
                vfn = special.wofz(z).real
                v_profile[:,i]=(2*np.sqrt(ln2/pi))*(vfn/Lg)
            if num_param==5:
                v_profile[:,i]=s*(v_profile[:,i]/np.max(v_profile[:,i]))+ offset
            elif num_param==6:
                v_profile[:,i]=s*(v_profile[:,i]/np.max(v_profile[:,i]))+(slope*lam+offset)
            else:
                v_profile[:,i]=s*(v_profile[:,i]/np.max(v_profile[:,i])) 
            if mask is not None:
                pass  # not implemented yet
        return np.sum(v_profile,axis=1)
    
    def sse1 (param,lam,y,AmpConst,WveConst,Diff_space,numparam=4):
        #objective function
        param =param_transform(param)
        if Diff_space:
            ynew = diff_umass(lam,funeval(param,lam,AmpConst,WveConst,num_param=numparam),order=2).flatten()
            chi =np.sqrt(np.abs(y.reshape(-1,1)-ynew.reshape(-1,1)))
            for h in range(len(WveConst)):
                b=np.argmax(lam==WveConst[h])
                chi[b] = chi[b]*10
        else:
            ynew = funeval(param,lam,AmpConst,WveConst,num_param=numparam)
            dif_y = diff_umass(lam,y[:,0],order=2)
            dif_ynew = diff_umass(lam,ynew,order=2)
            mu = 0.6
            chi1 = (1-mu)*(np.sqrt(np.abs(dif_y-dif_ynew))) 
            chi2=(mu)*(np.sqrt(np.abs(y[:,1].reshape(-1,1)-ynew.reshape(-1,1))))
            for u in range(len(chi2)):
                if abs(chi2[u]) >0.01:
                        chi2[u] =chi2[u]*10
            chi =chi1+chi2
        return np.sqrt(chi)
    
    def sse2(param,lam,AmpConst,WveConst,Diff_space,numparam=4,mask=None):
             if Diff_space:
                 ynew = diff_umass(lam,funeval(param,lam,AmpConst,WveConst,num_param=numparam,mask=mask),order=2).flatten()
             else:
                 ynew = funeval(param,lam,AmpConst,WveConst,num_param=numparam,mask=mask)
             return ynew
                 
             
    def vary_or_fix(i,numpar =None,tofix=None, pk=True):
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
                    return not(((i-numpar)%numpar) in np.array(tofix))
    
    
    def get_fitted_params(r,numPar):
        # retrieve fitted parameters from lmfit results
        params = np.zeros(numPar)
        for i in range(numPar):
            params[i]=r.params['p'+str(i)].value
        return params
    
    def range_for_fit(lam,y,peak_value,near_pks,wav=5):
        # limit range of data used for fitting to speed the fitting process
        minW = np.max(np.array([np.min(near_pks)-(wav),np.min(lam)]))
        maxW = np.min(np.array([np.max(near_pks)+(wav),np.max(lam)]))
        srt1 = lam>=minW
        lam =lam[srt1]
        y =y[srt1]
        srt2 = lam<=maxW
        lam =lam[srt2]      
        y =y[srt2]
        return lam,y
    
    def my_plot(Diff_space):
        from matplotlib import gridspec
        if Diff_space:
            plt.figure(num=4, figsize=(4,3))
        else:
            plt.figure(num=44, figsize=(4,3))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
        plt.subplot(gs[0])
        plt.plot(lam,data,'r',lw=1.5, label='actual')
        plt.plot(lam,diffynew,'b',lw=1.5,label='fit')
     
        if Diff_space:
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
        if Diff_space:
            plt.ylim(-0.5,0.5)
        else:
            plt.ylim(-0.5,0.5)
        plt.xlim(min(lam),max(lam))
        plt.ylabel('res.')
        plt.tight_layout()
    
    
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
        
    def my_plot2(Diff_space):
        if Diff_space:
            plt.figure(num=5, figsize=(4,3))
        else:
            plt.figure(num=55, figsize=(4,3))
        plt.ylabel(r'Intensity $\times 10^\mathrm{12}$ (arb. units)') 
        plt.xlabel('Wavelength (nm)')
        plt.ylim(0,max((data*max_data)/1e12))
        plt.tight_layout()
    
    if Diff_space:
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
    _,_,_,pred=LinearWeightedRegression(lam,data,tau,lam)
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
            pksin,_ = find_peaks(ycwt, prominence=0.0)
            pksx = lam[pksin].reshape(-1,1)
            pksy = ycwt[pksin].reshape(-1,1)
        
    sorting = np.argsort(pksx.flatten())# sorting uneccesary for automatic but may be for manual peak selection
    pksx = pksx[sorting] 
    pksy = pksy[sorting]
    pred = pred.reshape(-1,1)
    AmpConst = pksy #Amplitude constant
    WveConst = pksx #Wavelength constant
    if Diff_space:
        plt.figure(num=1,figsize=(4,3), dpi=my_dpi)
        plt.plot(lam,data,'r',lw=1.5, label='spectrum')
        plt.plot(pksx,pksy,'k.',markersize=5, label='peak')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r' Norm. intensity')
        plt.tight_layout()
        plt.xlim(min(lam),max(lam))
        plt.ylim(0,max(data))
        plt.show()
        plt.savefig(Name1+'.svg', format='svg')
        plt.savefig(Name1+'.tiff',dpi=my_dpi)
        
        
        plt.figure(num=2,figsize=(4,3))       
        plt.plot(lam,(data*max_data)/1e12,'r',lw=1.5, label='raw')
        plt.plot(lam,(pred*max_data)/1e12,'b',lw=1.5,label='smooth')     
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r'Intensity $\times 10^\mathrm{12}$ (arb. units)')  
        plt.tight_layout()
        plt.xlim(min(lam),max(lam))
        plt.ylim(0,max((data*max_data)/1e12))
        plt.show()
        plt.savefig(Name2+'.svg', format='svg', dpi=my_dpi)
        plt.savefig(Name2+'.tiff',dpi=my_dpi)

    data_t =np.zeros((len(lam),2))
    data_t[:,0] = lam
    data_t[:,1] = pred.flatten()
    if Diff_space is True:
        raw_data = np.column_stack((data_t[:,0],diff_umass(data_t[:,0], data_t[:,1],order=2))) 
    else:
        raw_data = data_t
    pks_sortVal =np.ones((numparam,len(pksx)),dtype=int)
    lam_mask =np.zeros((len(lam),len(pksx))) 
    if len(prior['paramVal']) == 0:
       paramVal =np.zeros((numparam,len(pksx))) 
       paramVal[0,:] = 0.001
       paramVal[1,:] = 0.5
       paramVal[2,:] = 0.1
       paramVal[3,:] = 0.1
       if numparam==5:
           paramVal[4,:] = 0.01
       if numparam==6:
           paramVal[5,:] = 0.01
       
       lbVal =np.zeros((numparam,len(pksx))) 
       lbVal[0,:] = -0.5
       if numparam==6:
           lbVal[5,:] = -10.0

       
       ubVal =np.zeros((numparam,len(pksx))) 
       ubVal[0,:] = 0.5
       ubVal[1,:] = 1.0
       ubVal[2,:] = 1.0
       ubVal[3,:] = 5.0
       if numparam==5:
           ubVal[4,:] = 1.0
       if numparam==6:
           ubVal[4,:] = 1.0
           ubVal[5,:] = 10.0
    else:
        paramVal = prior['paramVal']
        lbVal = prior['lbVal']
        ubVal = prior['ubVal']
    start = time.time()
    for k in range(len(pksx)):  
        if Diff_space is True:   
            param=       paramVal[:,max(0,k-(diff_peaks-1)):min(len(pksy),k+diff_peaks)]#   % Positions
            pks_sort= pks_sortVal[:,max(0,k-(diff_peaks-1)):min(len(pksy),k+diff_peaks)]#   % Positions
            lb =            lbVal[:,max(0,k-(diff_peaks-1)):min(len(pksy),k+diff_peaks)]
            ub =            ubVal[:,max(0,k-(diff_peaks-1)):min(len(pksy),k+diff_peaks)]
        else:
            param=       paramVal[:,max(0,k-(raw_peaks-1)):min(len(pksy),k+raw_peaks)]#   % Positions
            pks_sort= pks_sortVal[:,max(0,k-(raw_peaks-1)):min(len(pksy),k+raw_peaks)]#   % Positions
            lb =            lbVal[:,max(0,k-(raw_peaks-1)):min(len(pksy),k+raw_peaks)]
            ub =            ubVal[:,max(0,k-(raw_peaks-1)):min(len(pksy),k+raw_peaks)]

        y = raw_data[:,1]
        fit_params = Parameters()
        param = param.T.reshape(-1,1).flatten()
        pks_sort = pks_sort.T.reshape(-1,1).flatten()
        ub = ub.T.reshape(-1,1).flatten()
        lb = lb.T.reshape(-1,1).flatten()
        
        if Diff_space is True:
            mnk = np.max([0,k-(diff_peaks-1)])
            msk = np.min([len(pksy),k+diff_peaks])
            for i in range(len(param)):
                fit_params.add('p'+str(i), value=param[i], max=ub[i], min=lb[i], vary=vary_or_fix(i,numpar =numparam,tofix=None,pk=bool(pks_sort[i])))
           
            trunc_lam,trunc_y = range_for_fit(lam,y,pksx[k],WveConst[mnk:msk],wav=trunc_wav)
            p_min = minimize(sse1,fit_params,method='least_squares',args=(trunc_lam,trunc_y,AmpConst[mnk:msk],WveConst[mnk:msk],Diff_space,numparam,))
            p_min = get_fitted_params(p_min,len(param)).reshape((numparam,-1),order='F')
            if (p_min.shape[1]<((diff_peaks*2)-1) and k<(diff_peaks-1)):
                paramVal[:,k]= p_min[:,k]
            else:
                paramVal[:,k]= p_min[:,(diff_peaks-1)]                 
            pks_sortVal[:,k]= 0          
        else:
            mnk = np.max([0,k-(raw_peaks-1)])
            msk = np.min([len(pksy),k+raw_peaks])
            for i in range(len(param)):
                # pdb.set_trace()
                fit_params.add('p'+str(i), value=param[i], max=ub[i], min=lb[i], vary=vary_or_fix(i,numpar=numparam,tofix=None,pk=bool(pks_sort[i])))
            trunc_lam,trunc_y = range_for_fit(lam,y,pksx[k],WveConst[mnk:msk],wav=trunc_wav)
            _,trunc_d = range_for_fit(lam,data,pksx[k],WveConst[mnk:msk],wav=trunc_wav)
            new_y = np.hstack((trunc_y.reshape(-1,1),trunc_d.reshape(-1,1)))

            p_min = minimize(sse1,fit_params,method='least_squares',args=(trunc_lam,new_y,AmpConst[mnk:msk],WveConst[mnk:msk],Diff_space,numparam,))
            p_min = get_fitted_params(p_min,len(param)).reshape((numparam,-1),order='F')
            if (p_min.shape[1]<((raw_peaks*2)-1)  and k<(raw_peaks-1)):
                try:
                    paramVal[:,k]= p_min[:,k]
                except IndexError:
                    pdb.set_trace()
            else:
                paramVal[:,k]= p_min[:,(raw_peaks-1)]      
            pks_sortVal[:,k]= 0
        mg = np.zeros_like(lam).copy()
        magic_lam=lam.searchsorted(trunc_lam) # wavelength selector for wavelength used in fitting
        mg[magic_lam]=1.0
        lam_mask[:,k] = mg

   
    diffynew = sse2(paramVal,lam,AmpConst,WveConst,False,numparam=numparam,mask=lam_mask)
    ynew2  = funeval(paramVal,lam,AmpConst,WveConst,num_param=numparam, mask=lam_mask)
    lam4 =np.arange(np.min(lam),np.max(lam),0.001)
    ynew4 = funeval(paramVal,lam4,AmpConst,WveConst,num_param=numparam,mask=lam_mask)
    stop = time.time()
    print(f'time taken:{(stop-start)} s')


    my_plot(Diff_space)
    plt.savefig(Name3+'.tiff',dpi=my_dpi)
    plt.savefig(Name3+'.svg', format='svg', dpi=my_dpi)

    my_plot2(Diff_space)
    integ = np.zeros((paramVal.shape[1],5))
    for t in range(paramVal.shape[1]):
        p_min  = paramVal[:,t]
        fwhm= 0.5346*p_min[3]+np.sqrt(0.2166*p_min[3]**2+p_min[2]**2)# Voigt FWHM
        p_min2 = WveConst[t]-p_min[0]
        lam2 = np.arange(p_min2-20*fwhm,p_min2+20*fwhm,0.01).flatten()
        ynew3 = funeval(p_min,lam2,AmpConst[t],WveConst[t],num_param=numparam)
        integ[t,0] =p_min2
        integ[t,1] = p_min[1]*AmpConst[t]
        integ[t,2]= p_min[2]
        integ[t,3] = p_min[3]
        integ[t,4] =np.trapz(lam2)*max_data
        plt.plot(lam2,(ynew3*max_data)/1e12,color='green',linestyle='--',lw=1)
    plt.plot(lam2,(ynew3*max_data)/1e12,color='green',linestyle='--',lw=1, label='profile') # hact to give the individua peaks label    
    plt.plot(lam,(data*max_data)/1e12,'r-',lw=1.5, label='actual')
    plt.plot(lam4,(ynew4*max_data)/1e12,'b-',lw=1.5,label='sum of profiles')
    plt.xlim(min(lam),max(lam))
    plt.tight_layout()
    plt.savefig(Name4+'.svg', format='svg', dpi=my_dpi)
    plt.savefig(Name4+'.tiff',dpi=my_dpi)
    if max(lam)<400:
        plt.xlim(258,264)
        plt.ylim(0,4.5)
    if max(lam)>400 and max(lam)<900:
        plt.xlim(392,398)
    if  max(lam)>700:
        plt.xlim(760,785)

    plt.savefig(Name4+'zoom.svg', format='svg', dpi=my_dpi)
    plt.savefig(Name4+'zoom.tiff',dpi=my_dpi)
    if Diff_space:
        my_plot_diff()
        plt.savefig(Name5+'zoom.svg', format='svg', dpi=my_dpi)
        plt.savefig(Name5+'zoom.tiff',dpi=my_dpi)
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
    
    

        
        
        
    
    
    
