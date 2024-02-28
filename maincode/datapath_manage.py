# all audio/image data are all prepossed by stft/encoder 

dataset = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/'
speaker_list_t = list(range(1,3))#how many speakers in total for train
number_list_t = list(range(1, 4))# how many samples for one speaker

speaker_list_v = list(range(1,2))#how many speakers in total for val
number_list_v = list(range(1, 2))# how many samples for one speaker

speaker_list_s = list(range(1,2))#how many speakers in total for test
number_list_s = list(range(1, 2))# how many samples for one speaker


#-----train path -----
def train_list(dataset):
    speaker_path_t = [dataset + 'train_data/' + 'Pt' + str(i).zfill(2) + '/' for i in speaker_list_t]
    temp_list_t = []
    count_t = 0
    for a in speaker_path_t:
        count_t = count_t + 1
        for b in number_list_t:
            temp_list_t.extend([(a + 'clean/c_P'+ str(count_t)+ '_' + str(b) + '.pt', a + 'noisy/n_P' + str(count_t)+ '_' + str(b) + '.pt', a + 'image/i_P' +str(count_t)+ '_'+ str(b) + '.pt')])
    return temp_list_t 

#----- validation path -----
def val_list(dataset):
    speaker_path_v = [dataset + 'val_data/' + 'Pv' + str(i).zfill(2) + '/' for i in speaker_list_v]
    temp_list_v = []
    count_v = 0
    for a in speaker_path_v:
        count_v =count_v + 1
        for b in number_list_v:
            temp_list_v.extend([(a + 'clean/vc_P'+ str(count_v)+ '_' + str(b) + '.pt', a + 'noisy/vn_P' + str(count_v)+ '_' + str(b) + '.pt', a + 'image/vi_P' + str(count_v)+ '_' + str(b) + '.pt')])
    return temp_list_v

#----- test path -----
def test_list(dataset):
    speaker_path_s = [dataset + 'test_data/' + 'Ps' + str(i).zfill(2) + '/' for i in speaker_list_s]
    temp_list_s = []
    count_s = 0
    for a in speaker_path_s:
        count_s = count_s + 1
        for b in number_list_s:
            temp_list_s.extend([(a + 'clean/sc_P'+ str(count_s)+ '_' +str(b) + '.pt', a + 'noisy/sn_P' + str(count_s)+ '_' +str(b) + '.pt', a + 'image/si_P' + str(count_s)+ '_' +str(b) + '.pt')])
    return temp_list_s


 
def datapath_manage(dataset): 
    t_datapath = train_list(dataset) 
    v_datapath = val_list(dataset)
    s_datapath = test_list(dataset)
    

    return t_datapath, v_datapath, s_datapath

