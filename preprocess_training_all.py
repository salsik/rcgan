import os
import time
import argparse
import numpy as np
import pickle

# Custom Classes
import preprocess


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_for_training(train_A_dir, cache_folder):
    num_mcep = 24
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128

    print("Starting to prepocess data.......")
    start_time = time.time()

    wavs_A = preprocess.load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    #wavs_B = preprocess.load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
        wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    #f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = preprocess.world_encode_data(
     #   wave=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)




    username= os.path.basename(os.path.normpath(train_A_dir))


    print("Log Pitch : "+username)
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))

    coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)


    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)


    # we can split them and we can put them together
    np.savez(os.path.join(cache_folder, 'logf0s_mcep_normalization_'+username +'.npz'),
             mean_logf0=log_f0s_mean_A,
             std_logf0=log_f0s_std_A,
             mean_mcep=coded_sps_A_mean,
             std_mcep=coded_sps_A_std)

    #np.savez(os.path.join(cache_folder, 'mcep_normalization_'+username +'.npz'),
     #        mean_A=coded_sps_A_mean,
      #       std_A=coded_sps_A_std)


    save_pickle(variable=coded_sps_A_norm,
                fileName=os.path.join(cache_folder, "coded_sps_norm_"+username +".pickle"))
    #save_pickle(variable=coded_sps_B_norm,
     #           fileName=os.path.join(cache_folder, "coded_sps_B_norm.pickle"))

    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache_all for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    train_A_dir_default = '../data/vcc2016_training/SF2/'
    train_B_dir_default = '../data/vcc2016_training/SM2/'

    cache_folder_default = '../cache_all/'


    parser.add_argument('--train_A_dir', type=str,
                        help="Directory for source voice sample", default=train_A_dir_default)
    parser.add_argument('--cache_folder', type=str,
                        help="Store preprocessed data in cache folders", default=cache_folder_default)
    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir

    cache_folder = argv.cache_folder

    train_A_dir_arr = []
    train_A_dir_arr.append('../data/vcc2016_training/SF1/')
    train_A_dir_arr.append('../data/vcc2016_training/SF2/')
    train_A_dir_arr.append('../data/vcc2016_training/SF3/')
    train_A_dir_arr.append('../data/vcc2016_training/SM1/')
    train_A_dir_arr.append('../data/vcc2016_training/SM2/')
    train_A_dir_arr.append('../data/vcc2016_training/TF1/')
    train_A_dir_arr.append('../data/vcc2016_training/TF2/')
    train_A_dir_arr.append('../data/vcc2016_training/TM1/')
    train_A_dir_arr.append('../data/vcc2016_training/TM2/')
    train_A_dir_arr.append('../data/vcc2016_training/TM3/')


    for train_dir in train_A_dir_arr:
        preprocess_for_training(train_dir, cache_folder)
