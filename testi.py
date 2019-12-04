import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle

import preprocess
from trainingDataset import trainingDataset
from model_GLU import Generator, Discriminator



def loadPickleFile(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)



parser = argparse.ArgumentParser(
description="Train CycleGAN using source dataset and target dataset")

logf0s_normalization_default = '../cache/logf0s_normalization.npz'
mcep_normalization_default = '../cache/mcep_normalization.npz'
coded_sps_A_norm = '../cache/coded_sps_A_norm.pickle'
coded_sps_B_norm = '../cache/coded_sps_B_norm.pickle'
model_checkpoint = '../cache/model_checkpoint/'
resume_training_at = '../cache/model_checkpoint/_CycleGAN_CheckPoint'
resume_training_at = None

validation_A_dir_default = '../data/vcc2016_training/evaluation_all/SF1/'
output_A_dir_default = '../data/vcc2016_training/converted_sound/SF1'

validation_B_dir_default = '../data/vcc2016_training/evaluation_all/TF2/'
output_B_dir_default = '../data/vcc2016_training/converted_sound/TF2/'

parser.add_argument('--logf0s_normalization', type=str,
        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
parser.add_argument('--mcep_normalization', type=str,
        help="Cached location for mcep normalization", default=mcep_normalization_default)
parser.add_argument('--coded_sps_A_norm', type=str,
        help="mcep norm for data A", default=coded_sps_A_norm)
parser.add_argument('--coded_sps_B_norm', type=str,
        help="mcep norm for data B", default=coded_sps_B_norm)
parser.add_argument('--model_checkpoint', type=str,
        help="location where you want to save the model", default=model_checkpoint)
parser.add_argument('--resume_training_at', type=str,
        help="Location of the pre-trained model to resume training",
        default=resume_training_at)
parser.add_argument('--validation_A_dir', type=str,
        help="validation set for sound source A", default=validation_A_dir_default)
parser.add_argument('--output_A_dir', type=str,
        help="output for converted Sound Source A", default=output_A_dir_default)
parser.add_argument('--validation_B_dir', type=str,
        help="Validation set for sound source B", default=validation_B_dir_default)
parser.add_argument('--output_B_dir', type=str,
        help="Output for converted sound Source B", default=output_B_dir_default)

argv = parser.parse_args()


logf0s_normalization = argv.logf0s_normalization
mcep_normalization = argv.mcep_normalization
coded_sps_A_norm = argv.coded_sps_A_norm
coded_sps_B_norm = argv.coded_sps_B_norm
model_checkpoint = argv.model_checkpoint
resume_training_at = argv.resume_training_at

validation_A_dir = argv.validation_A_dir
output_A_dir = argv.output_A_dir
validation_B_dir = argv.validation_B_dir
output_B_dir = argv.output_B_dir

restart_training_at=resume_training_at


start_epoch = 0
num_epochs = 5000
mini_batch_size = 1
dataset_A = loadPickleFile(coded_sps_A_norm)
dataset_B = loadPickleFile(coded_sps_B_norm)
device = torch.device(
'cuda' if torch.cuda.is_available() else 'cpu')

# Speech Parameters
logf0s_normalization = np.load(logf0s_normalization)
log_f0s_mean_A = logf0s_normalization['mean_A']
log_f0s_std_A = logf0s_normalization['std_A']
log_f0s_mean_B = logf0s_normalization['mean_B']
log_f0s_std_B = logf0s_normalization['std_B']

mcep_normalization = np.load(mcep_normalization)
coded_sps_A_mean = mcep_normalization['mean_A']
coded_sps_A_std = mcep_normalization['std_A']
coded_sps_B_mean = mcep_normalization['mean_B']
coded_sps_B_std = mcep_normalization['std_B']





 # Generator and Discriminator
generator_A2B = Generator().to(device)
generator_B2A = Generator().to(device)
discriminator_A = Discriminator().to(device)
discriminator_B = Discriminator().to(device)

# Loss Functions
criterion_mse = torch.nn.MSELoss()

# Optimizer

g_params = list(generator_A2B.parameters()) + \
    list(generator_B2A.parameters())
    
g_params1 = list(generator_A2B.parameters()) + list(generator_B2A.parameters())

d_params = list(discriminator_A.parameters()) +   list(discriminator_B.parameters())

# Initial learning rates
generator_lr = 0.0002
discriminator_lr = 0.0001

# Learning rate decay
generator_lr_decay = generator_lr / 200000
discriminator_lr_decay = discriminator_lr / 200000

# Starts learning rate decay from after this many iterations have passed
start_decay = 200000

generator_optimizer = torch.optim.Adam(
    g_params, lr=generator_lr, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(
    d_params, lr=discriminator_lr, betas=(0.5, 0.999))

# To Load save previously saved models
modelCheckpoint = model_checkpoint

# Validation set Parameters
validation_A_dir = validation_A_dir
output_A_dir = output_A_dir
validation_B_dir = validation_B_dir
output_B_dir = output_B_dir

# Storing Discriminatior and Generator Loss
generator_loss_store = []
discriminator_loss_store = []

file_name = 'log_store_non_sigmoid.txt'

if restart_training_at is not None:
    # Training will resume from previous checkpoint
    start_epoch = loadModel(restart_training_at)
    print("Training resumed")



###################################################################
    
                    # finish initialization and start training
    
###################################################################



# 1 loop of training
                    
                    

start_time_epoch = time.time()

# Constants
cycle_loss_lambda = 10
identity_loss_lambda = 5

# Preparing Dataset
n_samples = len(dataset_A)

dataset = trainingDataset(datasetA=dataset_A,
                  datasetB=dataset_B,
                  n_frames=128)
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                   batch_size=mini_batch_size,
                                   shuffle=True,
                                   drop_last=False)


for i, (real_A, real_B) in enumerate(train_loader):
    print (i,len(real_A),len(real_B),real_B)














