from lsst.daf.butler import Butler
from astropy.table import Table
from tqdm import tqdm
import numpy as np
from copy import copy

butler = Butler('/repo/main')

fname = 'u_scichris_aosBaseline_danish1_donutQualityTable_N_donuts_all_det_select_good_new.txt'
donutQualityTableSel = Table.read(fname, format='ascii')

selection = copy(donutQualityTableSel) # donutQualityTableNdonuts[donutQualityTableNdonuts['Ndonuts'] > 15]

# loading the results for N visits 
visits_unique = np.unique(selection['visit']) 
# these are skipped because results didn't work for all detectors for all methods 
skip_visit =[2024110300085, 2024120200150, 2024120600341, 2024121000362, 2024120100423]
visits = visits_unique[~np.in1d(visits_unique,  skip_visit)]
results_visit = {}

Nvisits = len(visits)
for visit  in tqdm(visits[:Nvisits], desc="Processing"):
    
    results_visit_danish_bin1 = {}
    results_visit_danish_bin2 = {}
    results_visit_danish_bin4 = {}
    results_visit_tie_bin1 = {}
    results_visit_tie_bin2 = {}
    results_visit_tie_bin4 = {}
    for detector in range(9):
        
        dataId = {'instrument':'LSSTComCam', 'detector':detector, 
                'visit':visit
             }
    
        results_visit_tie_bin1[detector] = butler.get('zernikes', dataId = dataId, 
                                                      collections=['u/scichris/aosBaseline_tie_binning_1'])
        results_visit_tie_bin2[detector] = butler.get('zernikes', dataId = dataId, 
                                                      collections=['u/scichris/aosBaseline_tie_binning_2'])
        results_visit_tie_bin4[detector] = butler.get('zernikes', dataId = dataId, 
                                                      collections=['u/scichris/aosBaseline_tie_binning_4'])
        results_visit_danish_bin1[detector] = butler.get('zernikes', dataId = dataId,  
                                                         collections=['u/scichris/aosBaseline_danish_binning_1'])
        results_visit_danish_bin2[detector] = butler.get('zernikes', dataId = dataId,  
                                                         collections=['u/scichris/aosBaseline_danish_binning_2'])
        results_visit_danish_bin4[detector] = butler.get('zernikes', dataId = dataId,  
                                                         collections=['u/scichris/aosBaseline_danish_binning_4'])
    results_visit[visit] = {'tie1':results_visit_tie_bin1, 
                            'tie2':results_visit_tie_bin2,
                            'tie4':results_visit_tie_bin4, 
                            'danish1':results_visit_danish_bin1,
                            'danish2':results_visit_danish_bin2, 
                            'danish4':results_visit_danish_bin4,
                           }
file = f'u_scichris_aosBaseline_tie_danish_zernikes_tables_{Nvisits}.npy'
np.save(file, results_visit, allow_pickle=True)