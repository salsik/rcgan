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


def preprocess_for_training(train_dirs, cache_folder):
    num_mcep = 24
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128

    print("Starting to prepocess data.......")
    start_time = time.time()


    #wavs_A = preprocess.load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    #wavs_B = preprocess.load_wavs(wav_dir=train_B_dir, sr=sampling_rate)




    log_f0s_mean_s = []
    log_f0s_std_s = []
    coded_sps_s_norm =[]
    coded_sps_s_mean = []
    coded_sps_s_std = []

    for dir in train_dirs:
        waves0=preprocess.load_wavs(wav_dir=dir, sr=sampling_rate)
        #waves.append(waves0)
        f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
            wave=waves0, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

        #f0s_s.append(f0s_A)
        #timeaxes_s.append(timeaxes_A)
        #sps_s.append(sps_A)
        #aps_s.append(aps_A)
        #coded_sps_s.append(coded_sps_A)

        log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)
        log_f0s_mean_s.append(log_f0s_mean_A)
        log_f0s_std_s.append(log_f0s_std_A)

        print("Log Pitch user number ",str(len(log_f0s_std_s)))
        print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))

        coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)
        #coded_sps_s_transposed.append(coded_sps_A_transposed)

        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
            coded_sps=coded_sps_A_transposed)
        coded_sps_s_norm.append(coded_sps_A_norm)
        coded_sps_s_mean.append(coded_sps_A_mean)
        coded_sps_s_std.append(coded_sps_A_std)



    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)


    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
     mean_s=log_f0s_mean_s,
     std_s=log_f0s_std_s)


    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean_s=coded_sps_s_mean,
             std_s=coded_sps_s_std)

    #save aslso pickles variable here
    np.savez(os.path.join(cache_folder, 'mcep_normalization_withnorm.npz'),coded_sps_s_norm=coded_sps_s_norm)



    for i in range (len(coded_sps_s_norm)):
        save_pickle(variable=coded_sps_s_norm[i],
                fileName=os.path.join(cache_folder, "coded_sps_s_norm"+str(i)+".pickle"))

    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache_RCGAN for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))


    #    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
     #   wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
 #   f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = preprocess.world_encode_data(
    #    wave=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)


#    log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)
#    log_f0s_mean_B, log_f0s_std_B = preprocess.logf0_statistics(f0s=f0s_B)

  #  print("Log Pitch A")
  #  print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))
  #  print("Log Pitch B")
   # print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_B, log_f0s_std_B))

  #  coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)
  #  coded_sps_B_transposed = preprocess.transpose_in_list(lst=coded_sps_B)

  #  coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
   #     coded_sps=coded_sps_A_transposed)
  #  coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = preprocess.coded_sps_normalization_fit_transform(
   #     coded_sps=coded_sps_B_transposed)



  #  np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
   #          mean_A=log_f0s_mean_A,
    #         std_A=log_f0s_std_A,
    #         mean_B=log_f0s_mean_B,
     #        std_B=log_f0s_std_B)

   # np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
    #         mean_A=coded_sps_A_mean,
     #        std_A=coded_sps_A_std,
      #       mean_B=coded_sps_B_mean,
       #      std_B=coded_sps_B_std)

   # save_pickle(variable=coded_sps_A_norm,
   #             fileName=os.path.join(cache_folder, "coded_sps_A_norm.pickle"))
   # save_pickle(variable=coded_sps_B_norm,
   #             fileName=os.path.join(cache_folder, "coded_sps_B_norm.pickle"))






if __name__ == '__main3__':
    import numpy as np

    arr1 = np.arange(8).reshape(2, 4)
    arr2 = np.arange(10).reshape(2, 5)
    arr3 = np.arange(9).reshape(3, 3)

    arr=[]

    arr.append(arr1)
    arr.append(arr2)
    arr.append(arr3)

    np.savez('mat.npz', name1=arr1, name2=arr2)

    np.savez('mat.npz',zbr4=arr)


    data = np.load('mat.npz')

    print (data['zbr4'][1])
    #print (data['name2'])
    #print (data['zbr4'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    train_A_dir_default = '../data/vcc2016_training/SF2/'
    train_B_dir_default = '../data/vcc2016_training/SM2/'
    train_C_dir_default = '../data/vcc2016_training/TM3/'
    train_D_dir_default = '../data/vcc2016_training/TF1/'
    cache_folder_default = '../cache_RCGan/'

    parser.add_argument('--train_A_dir', type=str,
                        help="Directory for source voice sample", default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str,
                        help="Directory for target voice sample", default=train_B_dir_default)
    parser.add_argument('--cache_folder', type=str,
                        help="Store preprocessed data in cache folders", default=cache_folder_default)
    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    cache_folder = argv.cache_folder


    train_dirs =[]
    train_dirs.append(train_A_dir)
    train_dirs.append(train_B_dir)
    train_dirs.append(train_C_dir_default)
    train_dirs.append(train_D_dir_default)

    preprocess_for_training(train_dirs, cache_folder)
