def Auto_fit(peak,same_for_all,WaveR,tau,data,Diff_space,prior,save_name,sample_no,num_param=4,count=0,pkfall={'x':0,'y':0}):
    import pymsgbox as pmb
    import numpy as np
    import pdb
    from matplotlib import pyplot as plt
    from Extractfunction import EF
    # This programs identifies and fits peaks in a spectrum in an automatic manner
    # Input:
    # peak      -  % manual %autorep ,identify peak automatically (differentiation) or manually
    #same_for_all- % Use same peak positions for subsequent spectra for more that one spectrum
    #WaveR      -  % Wavelength range [minWave,maxWave]
    #threshVal  -  % percent threshold above which peaks will be determined (0-1)
    #DnC        -  % Divide and concquer ('yes to use)
    #DnC_peaks  -  % Peak to fit at a time but only one peak will be considered %(see manual for further expalanation)
    #tau        -  % smoothening factor for spectrum small value- less smoothening
    # Ebo Ewusi-Annan, July 2020, UMASS Lowell python version 
    
    # peak = 'autorep';% manual %autorep identify peak automatically (differentiation) or manually
    # same_for_all = 0;% Use same peak positions for subsequent spectra for more that one spectrum
    # tau=0.03;        % smoothening facor
    # parallel = 0;    % use parellel to do fitting % 1 -paralell or any character for - no parallel
    minWave =WaveR[0]    # minimum wavlength value
    maxWave =WaveR[1];    # maximum wavelength value
    Cdata = data['features']
    x= data['x']
    assert(len(x)==Cdata.shape[1]),'Dimension of wavlength data is not equal to spectrum dimension'
    if min(x) > minWave:
        minWave = min(x)
        pmb.alert('Lower range reset to min of wavelength data','Input wavelength less than min wavelength data')
    if max(x) < maxWave:
        maxWave = max(x)
        pmb.alert('Upper range reset to max of wavelength data','Input wavelength greater than max wavelength data')
    x= x.reshape(-1,1)
    for kk in range(sample_no,sample_no+1):
        data1 = Cdata[kk,:]
        file1 = save_name   
        if Diff_space:
            intgName={'raw_smooth':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_raw_smooth',\
                      'peaks':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_figure_second_diff_peaks',\
                      'diff_fit':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_fit_compare_diff_fit',\
                      'diff_fit_peaks':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_fit_compare_diff_peaks',\
                       'diff_spec':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_diff_spec_compare'   }
        else:
            intgName={'fit':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_fit_compare_fit',\
                      'fit_peaks':file1+'_'+str(minWave)+'_'+str(maxWave)+'nm_fit_peaks'}           
        data1 = data1.reshape(-1,1)
        windo = (x>=minWave)&(x<=maxWave)
        x = x[windo]
        data1 = data1[windo]
        max_data = max(data1)
        data1 = data1/max_data
        data_in = np.column_stack((x,data1))
        # def image_creator(data1,num_scales= len(data1),wavelet_type='mexh',image_name='test.jpg',image_size=(299,299)):
        #     import cv2
        #     import pywt
        #     from matplotlib import pyplot as plt
        #     scales = np.arange(1,num_scales)
        #     coef, freqs = pywt.cwt(data1,scales,wavelet_type)
        #     def rescale(arr):
        #         arr_min = arr.min()
        #         arr_max = arr.max()
        #         return (arr - arr_min) / (arr_max - arr_min)
    
        #     arr = 255.0 * rescale(abs(coef))
        #     plt.figure(figsize=(15,10))
        #     plt.imshow(arr,cmap ='jet',interpolation='bilinear')
        #     plt.axis('off')
        #     plt.savefig(image_name, bbox_inches = 'tight', pad_inches = 0)
        #     plt.show()
        #     img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        #     return cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)
        # resized =image_creator(data1,num_scales= 250,wavelet_type='mexh',image_size=(299,299))
        # plt.figure(figsize=(15,10))
        # plt.imshow(resized)
        # plt.title('read image')
        # plt.show()
        # plt.figure(figsize=(15,10))
        # plt.plot(data1,'r')
        # plt.show()
       # pdb.set_trace()
        output = EF(data_in,tau,intgName,peak,count,pkfall,Diff_space,prior,minWave,maxWave,max_data,numparam =num_param)
        if same_for_all is True:
            count =1
        return output