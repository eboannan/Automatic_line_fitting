"""
Program to extract the profiles of lines from a spectrum
By Ebo Ewusi-Annan, July 2020, UMASS Lowell
"""
import numpy as np
import os
from Auto_peak_iden_fitting_67ChemCam import Auto_fit
import pickle
import pdb
import time
import warnings
warnings.filterwarnings('ignore')


peak = 'autorep' # autorep identify peak automatically (differentiation) or manual(incomplete implementation)
same_for_all = False;# Use same peak positions for subsequent spectra for more that one spectrum
WaveRegion = np.array([[380,390]]) # Wavelength regions to fit
tau=0.03        # smoothening factor for local weighted regression
Diff_space = True # %Fit in diff space - True or False

Spectra_name ='X_ChemCam_mean'
wavelength_file_name = 'Wavelength_ChemCam.csv'
location = '../../ChemCam'
Cdata = np.genfromtxt(location+'/'+Spectra_name+'.csv',delimiter=',')
x = np.genfromtxt('../../ChemCam/'+wavelength_file_name,delimiter=',')
data = {'features':Cdata,'x':x}

#create folders to store outputs
TempFolder = 'TempHold_py';
if  not os.path.isdir(location+'/'+TempFolder):
    os.mkdir(location+'/'+TempFolder)
    print('Created folder Temphold_py')
TempFigs = 'Figures_py';
if  not os.path.isdir(location+'/'+TempFigs):
    os.mkdir(location+'/'+TempFigs)
    print('Created folder Figures_py')
Results={}  # Results dictionary
start = time.time()
for m in range(Cdata.shape[0]):
    print(f'Processing sample {m+1}')
    Results[m]={}
    Results[m]['actual']={}
    Results[m]['fit']={}
    Results[m]['paramVal']={}
    Results[m]['pksx']={}
    Results[m]['pksy']={}
    Results[m]['integ']={}
    actual={}
    pksx={}
    pksy={}
    paramVal={}
    integ ={}
    fit ={}
    for r in range(WaveRegion.shape[0]):
        WaveR =WaveRegion[r,:]
        sample_no =m
        prior ={'lbVal':"",'ubVal':"",'paramVal':""}
        Diff_space=True
        save_name =location+'/'+TempFigs+'/'+str(sample_no)+'_'+Spectra_name;
        output =Auto_fit(peak,same_for_all,WaveR,tau,data,Diff_space,prior,save_name,sample_no)
        prior['lbVal'] = output['lbVal'];
        prior['ubVal'] = output['ubVal'] ;
        prior['paramVal'] = output['paramVal']    
        Diff_space =False    
        output =Auto_fit(peak,same_for_all,WaveR,tau,data,Diff_space,prior,save_name,sample_no)
        actual['actual'+'_WaveR_'+str(WaveR[0])+'_'+str(WaveR[1])+'nm']= output['actual']
        fit['fit'+'_WaveR_'+str(WaveR[0])+'_'+str(WaveR[1])+'nm']= output['fit']
        pksx['pksx'+'_WaveR_'+str(WaveR[0])+'_'+str(WaveR[1])+'nm']= output['pksx']
        pksy['pksy'+'_WaveR_'+str(WaveR[0])+'_'+str(WaveR[1])+'nm']= output['pksy']
        integ['Integ'+'_WaveR_'+str(WaveR[0])+'_'+str(WaveR[1])+'nm']= output['Integ']
        paramVal['paramVal'+'_WaveR_'+str(WaveR[0])+'_'+str(WaveR[1])+'nm']= output['paramVal']
    Results[m]['actual']=actual
    Results[m]['fit']=fit
    Results[m]['paramVal']=paramVal
    Results[m]['pksx']=pksx
    Results[m]['pksy']=pksy
    Results[m]['integ']=integ
f= open(location+"/"+TempFolder+"/Results.pkl","wb")
pickle.dump(Results,f)    # Save results as a pickle file
f.close()
stop = time.time()  
taken = ((stop-start)/60)/60   # Time taken to run program
print(f'Total time taken is {taken} hours ')