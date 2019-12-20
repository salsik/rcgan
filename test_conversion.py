"""

sys.path.append('C:\\Users\\atsumilab\\Desktop\\research\\code\\Voice-Conversion-RCGAN')
sys.path
import preprocess
%hist

"""

import torch

import numpy as np
import os
import librosa

import pickle

from preprocess import *


def loadPickleFile(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def get_coded_sp_norm (filePath,sampling_rate,frame_period,num_mcep,mean,std):


    wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)

    wav_original = wav


    # maybe we need here to do padding

    f0, timeaxis, sp, ap = preprocess.world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)

    coded_sp = preprocess.world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
    coded_sp_transposed = coded_sp.T
    #coded_sp_norm = (coded_sp_transposed - coded_sps_S_mean[from_]) / coded_sps_S_std[from_]

    coded_sp_norm = (coded_sp_transposed - mean) / std

    coded_sp_norm = np.array([coded_sp_norm])

    coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()


    return coded_sp_norm,f0,ap



## convert using tensor code
if __name__ == '__main__':



    logf0s_mcep_normalization_traineduser = '../cache_all/logf0s_mcep_normalization_SF2.npz'
    logf0s_mcep_normalization_newuser = '../cache_all/logf0s_mcep_normalization_SM2.npz'

    coded_sps_user_norm = '../cache_all/coded_sps_norm_SF2.pickle'
    coded_sps_newuser_norm = '../cache_all/coded_sps_norm_SM2.pickle'


    logf0s_mcep_norm_user = np.load(logf0s_mcep_normalization_traineduser)
    logf0s_mcep_norm_newuser = np.load(logf0s_mcep_normalization_newuser)

    log_f0s_mean_S = []
    log_f0s_std_S = []

    log_f0s_mean_S.append(logf0s_mcep_norm_user['mean_logf0'])
    log_f0s_mean_S.append(logf0s_mcep_norm_newuser['mean_logf0'])


    log_f0s_std_S.append(logf0s_mcep_norm_user['std_logf0'])
    log_f0s_std_S.append(logf0s_mcep_norm_newuser['std_logf0'])

    coded_sps_S_mean = []
    coded_sps_S_std = []


    coded_sps_S_mean.append(logf0s_mcep_norm_user['mean_mcep'])
    coded_sps_S_mean.append(logf0s_mcep_norm_newuser['mean_mcep'])


    coded_sps_S_std.append(logf0s_mcep_norm_user['std_mcep'])
    coded_sps_S_std.append(logf0s_mcep_norm_newuser['std_mcep'])




    num_mcep = 24
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128




    file = '200012.wav'

    validation_A_output_dir =  '../data/vcc2016_training/converted_sound_RCGAN/Converted_to_SM2/'


    validation_A_dir = '../data/vcc2016_training/evaluation_all/SF2'
    filepath = os.path.join(validation_A_dir, file)
    wav, _ = librosa.load(filepath, sr=sampling_rate, mono=True)
    #wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_S[0], std_log_src=log_f0s_std_S[0],
                                    mean_log_target=log_f0s_mean_S[1], std_log_target=log_f0s_std_S[1])

    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)

    #coded_sp_transposed = coded_sp.T
    #coded_sp_norm = (coded_sp_transposed - coded_sps_S_mean[0]) / coded_sps_S_std[0]

    coded_sp_converted = np.ascontiguousarray(coded_sp)
    decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)

    # insted of convert current norm to new norm we calculate again the coded sp norm for B
    # and for usre we don't do transpose and normalization ...
    #coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='A2B')[0]
    #coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
    #coded_sp_converted = coded_sp_converted.T

    file2 = '200003.wav'

    validation_B_dir = '../data/vcc2016_training/evaluation_all/SM2'
    filepath2 = os.path.join(validation_B_dir, file2)
    wav2, _ = librosa.load(filepath2, sr=sampling_rate, mono=True)
   # wav2 = wav_padding(wav=wav2, sr=sampling_rate, frame_period=frame_period, multiple=4)
    f0_2, timeaxis_2, sp_2, ap_2 = world_decompose(wav=wav2, fs=sampling_rate, frame_period=frame_period)
    f0_converted_2 = pitch_conversion(f0=f0_2, mean_log_src=log_f0s_mean_S[1], std_log_src=log_f0s_std_S[1],
                                      mean_log_target=log_f0s_mean_S[0], std_log_target=log_f0s_std_S[0])
    coded_sp_2 = world_encode_spectral_envelop(sp=sp_2, fs=sampling_rate, dim=num_mcep)

    #coded_sp_transposed2 = coded_sp_2.T

    #coded_sp_norm2 = (coded_sp_transposed2 - coded_sps_S_mean[1]) / coded_sps_S_std[1]

    #coded_sp_converted2 = coded_sp_norm2 * coded_sps_S_std[0] + coded_sps_S_mean[0]
    #coded_sp_converted2 = coded_sp_converted2.T

    if (coded_sp.shape[0] > coded_sp_2.shape[0]):
        ap = ap[0:coded_sp_2.shape[0], :]
        f0_converted = f0_converted[0:coded_sp_2.shape[0]]
    else:
        coded_sp_2 = coded_sp_2[0:coded_sp.shape[0], :]

    coded_sp_converted_2 = np.ascontiguousarray(coded_sp_2)
    decoded_sp_converted_2 = world_decode_spectral_envelop(coded_sp=coded_sp_converted_2, fs=sampling_rate)
    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted_2, ap=ap, fs=sampling_rate,
                                             frame_period=frame_period)
    librosa.output.write_wav(os.path.join(validation_A_output_dir, str(np.random.normal())+ os.path.basename(file)), wav_transformed,
                             sampling_rate)

# extend conversion to use pytorch
if __name__ == '__mainPY__':

    num_mcep = 24
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128

    from_ = 1
    to = 0
    persons = ["SF2", "SM2", "TM2"]  # this is the  order of inserting to lists

    validation_S_dir_default = '../data/vcc2016_training/evaluation_RCGAN/'
    output_S_dir_default = '../data/vcc2016_training/converted_sound_RCGAN/'
    validation_B_dir = validation_S_dir_default + "/" + persons[2] + "/"

    output_S_dir = '../data/vcc2016_training/converted_sound_RCGAN/'
    output_B_dir = output_S_dir + "/" + persons[from_] + "/"

    logf0s_mcep_normalization_zero = '../cache_all/logf0s_mcep_normalization_SF2.npz'
    logf0s_mcep_normalization_traineduser = '../cache_all/logf0s_mcep_normalization_SM2.npz'
    logf0s_mcep_normalization_newuser = '../cache_all/logf0s_mcep_normalization_TF2.npz'
    coded_sps_0_norm = '../cache_all/coded_sps_norm_SF2.pickle'
    coded_sps_user_norm = '../cache_all/coded_sps_norm_SM2.pickle'
    coded_sps_newuser_norm = '../cache_all/coded_sps_norm_TF2.pickle'

    logf0s_mcep_norm_zero = np.load(logf0s_mcep_normalization_zero)
    logf0s_mcep_norm_user = np.load(logf0s_mcep_normalization_traineduser)
    logf0s_mcep_norm_newuser = np.load(logf0s_mcep_normalization_newuser)

    log_f0s_mean_S = []
    log_f0s_std_S = []

    log_f0s_mean_S.append(logf0s_mcep_norm_zero['mean_logf0'])
    log_f0s_mean_S.append(logf0s_mcep_norm_user['mean_logf0'])
    log_f0s_mean_S.append(logf0s_mcep_norm_newuser['mean_logf0'])

    log_f0s_std_S.append(logf0s_mcep_norm_zero['std_logf0'])
    log_f0s_std_S.append(logf0s_mcep_norm_user['std_logf0'])
    log_f0s_std_S.append(logf0s_mcep_norm_newuser['std_logf0'])

    coded_sps_S_mean = []
    coded_sps_S_std = []

    coded_sps_S_mean.append(logf0s_mcep_norm_zero['mean_mcep'])
    coded_sps_S_mean.append(logf0s_mcep_norm_user['mean_mcep'])
    coded_sps_S_mean.append(logf0s_mcep_norm_newuser['mean_mcep'])

    coded_sps_S_std.append(logf0s_mcep_norm_zero['std_mcep'])
    coded_sps_S_std.append(logf0s_mcep_norm_user['std_mcep'])
    coded_sps_S_std.append(logf0s_mcep_norm_newuser['std_mcep'])

    file = '200001.wav'
    filePath = os.path.join(validation_B_dir, file)



    # this function do the whole thing to get coded_sp_norm for a wavefile instead of the down commented code
    coded_sp_norm, f0, ap = get_coded_sp_norm(filePath,sampling_rate,frame_period,num_mcep,coded_sps_S_mean[from_],coded_sps_S_std[from_])


    #wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
    #wav_original = wav



    #wav = preprocess.wav_padding(wav=wav,sr=sampling_rate,frame_period=frame_period,multiple=4)


    #f0, timeaxis, sp, ap = preprocess.world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
   # coded_sp = preprocess.world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
   # coded_sp_transposed = coded_sp.T
   # coded_sp_norm = (coded_sp_transposed - coded_sps_S_mean[from_]) / coded_sps_S_std[from_]

   # coded_sp_norm = np.array([coded_sp_norm])

   # coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()

    f0_converted = preprocess.pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_S[from_],
                                               std_log_src=log_f0s_std_S[from_],
                                               mean_log_target=log_f0s_mean_S[to],
                                               std_log_target=log_f0s_std_S[to])


    secondfile = '200001.wav'
    secondfilePath = os.path.join(validation_B_dir, secondfile)

    coded_sp_converted_norm, f0, _ap = get_coded_sp_norm(secondfilePath,sampling_rate,frame_period,num_mcep,coded_sps_S_mean[to],coded_sps_S_std[to])

    
    #coded_sp_converted_norm = decoder_List[to - 1](encoder_List[to - 1](coded_sp_norm))
    
    
    coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
    coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
    coded_sp_converted = coded_sp_converted_norm * \
                         coded_sps_S_std[to] + coded_sps_S_mean[to]
    coded_sp_converted = coded_sp_converted.T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decoded_sp_converted = preprocess.world_decode_spectral_envelop(
        coded_sp=coded_sp_converted, fs=sampling_rate)
    wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                        decoded_sp=decoded_sp_converted,
                                                        ap=ap,
                                                        fs=sampling_rate,
                                                        frame_period=frame_period)
    
    epoch =126
    librosa.output.write_wav(path=os.path.join(output_B_dir, "epoch_" + str(epoch) + "_" + os.path.basename(file)),
                             y=wav_transformed,
                             sr=sampling_rate)
    
    


