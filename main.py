import os
import numpy as np 
from main_process import main
from IOtools import txt_write


if __name__ == '__main__':
    
    
    opt = dict()

    dataset_list = {0:'SH_partA_Density_map',1:'SH_partB_Density_map'}
    model_list = {0:'model/SHA',1:'model/SHB'}
    max_num_list = {0:22,1:7}

    # step1: Create root path for dataset
    opt['num_workers'] = 0

    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1

    # --Network settinng    
    opt['psize'],opt['pstride'] = 64,64
    # -- start testing
    set_len = len(dataset_list)
    ti=0
    opt['dataset'] = dataset_list[ti]
    opt['trained_model_path'] = model_list[ti]
    opt['root_dir'] = os.path.join(r'Test_Data','SH_partA_Density_map')
    #-- set the max number and partition
    opt['max_num'] = max_num_list[ti]  
    partition_method = {0:'one_linear',1:'two_linear'}
    opt['partition'] = partition_method[1]
    opt['step'] = 0.5
    print('=='*36)
    print('Welcome to Gods Eye')
    '''
    import requests
    import json
    send_url = 'http://freegeoip.net/json'
    r = requests.get(send_url)
    j = json.loads(r.text)
    lat = j['latitude']
    lon = j['longitude']
    '''
    opt['location']=input('Enter the location: ')
    print('Gods Eye has begin:-')
    main(opt)