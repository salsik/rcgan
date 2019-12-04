import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle

import preprocess
from tensorflow.contrib.seq2seq import Decoder
from trainingDataset import trainingDataset

from trainingDataset import trainingRCGANDataset

from model_GLU import Generator, Discriminator


from model_RCGAN import Encoder, Decoder


class CycleGANTraining:
    def __init__(self,
                 logf0s_normalization,
                 mcep_normalization,
                 coded_sps_S_norm,
                 model_checkpoint,
                 validation_S_dir,
                 output_S_dir,
                 restart_training_at=None):
        self.start_epoch = 0
        self.num_epochs = 5 # this should be 5000 for the whole training
        self.mini_batch_size = 1


        self.datasets = []
        for dset in coded_sps_S_norm:
            self.datasets.append(self.loadPickleFile(dset))


        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Speech Parameters
        logf0s_normalization = np.load(logf0s_normalization)

        self.log_f0s_mean_S =[]
        self.log_f0s_std_S =[]

        for log in logf0s_normalization['mean_s']:
            self.log_f0s_mean_S.append(log)

        for log in logf0s_normalization['std_s']:
            self.log_f0s_std_S.append(log)

        #self.log_f0s_mean_A = logf0s_normalization['mean_A']
        #self.log_f0s_std_A = logf0s_normalization['std_A']
        #self.log_f0s_mean_B = logf0s_normalization['mean_B']
        #self.log_f0s_std_B = logf0s_normalization['std_B']

        mcep_normalization = np.load(mcep_normalization)

        self.coded_sps_S_mean =[]
        self.coded_sps_S_std = []
        for mcep in mcep_normalization['mean_s']:
            self.coded_sps_S_mean.append(mcep)

        for mcep in mcep_normalization['std_s']:
            self.coded_sps_S_std.append(mcep)

       #self.coded_sps_A_mean = mcep_normalization['mean_A']
        #self.coded_sps_A_std = mcep_normalization['std_A']
        #self.coded_sps_B_mean = mcep_normalization['mean_B']
        #self.coded_sps_B_std = mcep_normalization['std_B']

        # Generator and Discriminator
        #self.generator_A2B = Generator().to(self.device)
        #self.generator_B2A = Generator().to(self.device)
        #self.discriminator_A = Discriminator().to(self.device)
        #self.discriminator_B = Discriminator().to(self.device)

        # encoder and decoders are part of generators

        self.enc_0 = Encoder().to(self.device)
        self.dec_0 = Decoder().to(self.device)

        self.encoder_List = []
        # those names may change according to user name .. or the speaker name
        self.encoder_List.append(Encoder("Enc1").to(self.device))
        self.encoder_List.append(Encoder("Enc2").to(self.device))
        self.encoder_List.append(Encoder("Enc3").to(self.device))


        self.decoder_List = []

        self.decoder_List.append(Decoder("Dec1").to(self.device))
        self.decoder_List.append(Decoder("Dec2").to(self.device))
        self.decoder_List.append(Decoder("Dec3").to(self.device))


        # discriminator should be a list of discs related to decoders

        self.discriminator_0 = Discriminator().to(self.device)
        self.discriminator_List = []

        self.discriminator_List.append(Discriminator().to(self.device))
        self.discriminator_List.append(Discriminator().to(self.device))
        self.discriminator_List.append(Discriminator().to(self.device))


        # Loss Functions
        criterion_mse = torch.nn.MSELoss()

        # Optimizer
        g_params = list(self.enc_0.parameters())
        for enc in self.encoder_List:
            g_params += list(enc.parameters())

        g_params += list(self.dec_0.parameters())
        for dec in self.decoder_List:
            g_params += list(dec.parameters())

        d_params = list(self.discriminator_0.parameters())
        for disc in self.discriminator_List:
            d_params += list(disc.parameters())

        #d_params =list(self.discriminator_A.parameters()) + \
         #  list(self.discriminator_B.parameters())



        # Initial learning rates
        self.generator_lr = 0.0002
        self.discriminator_lr = 0.0001

        # Learning rate decay
        self.generator_lr_decay = self.generator_lr / 200000
        self.discriminator_lr_decay = self.discriminator_lr / 200000

        # Starts learning rate decay from after this many iterations have passed
        self.start_decay = 200000

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # To Load save previously saved models
        self.modelCheckpoint = model_checkpoint

        # Validation set Parameters
        self.validation_S_dir = validation_S_dir
        self.output_S_dir = output_S_dir
        #self.validation_B_dir = validation_B_dir
        #self.output_B_dir = output_B_dir

        # Storing Discriminatior and Generator Loss
        self.generator_loss_store = []
        self.discriminator_loss_store = []

        self.file_name = 'log_store_non_sigmoid.txt'


        # should reimplement load and save  again
        if restart_training_at is not None:
            # Training will resume from previous checkpoint
            self.start_epoch = self.loadModel(restart_training_at)
            print("Training resumed")

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        # Training Begins
        # from 0 to 5000 epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time_epoch = time.time()

            # Constants
            cycle_loss_lambda = 10
            identity_loss_lambda = 5
            radial_loss_lambda = 7


            # Preparing Dataset 162 voice sample as defined first
            n_samples = len(self.datasets[0])

            dataset = trainingRCGANDataset(self.datasets, n_frames=128)

          #  dataset = trainingDataset(datasetA=self.dataset_A,
           #                           datasetB=self.dataset_B,
            #                          n_frames=128)

            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.mini_batch_size,
                                                       shuffle=True,
                                                       drop_last=False)
            # real A is from the shape 1,24,128 and real B as well
            for i, real_data in enumerate(train_loader):

                num_iterations = (
                    n_samples // self.mini_batch_size) * epoch + i
                # print("iteration no: ", num_iterations, epoch)

                if num_iterations > 10000:
                    identity_loss_lambda = 0
                if num_iterations > self.start_decay:
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')


                # REAL 0  is the data we should pass it later to disc 0 after decoding 0
                real_0 = real_data[0].to(self.device).float()

                real_A = real_data[1].to(self.device).float()
                real_B = real_data[2].to(self.device).float()
                real_C = real_data[3].to(self.device).float()

                #real_A = real_A.to(self.device).float()
                #real_B = real_B.to(self.device).float()





                #start encoding

                encoded_A = self.encoder_List[0](real_A)
                encoded_B = self.encoder_List[1](real_B)
                encoded_C = self.encoder_List[2](real_C)

                #hidden features should be resblocks

            ###  is it a sum here or i think it should be an array of features for each user and then we make the radial loss through it
                hidden_features = encoded_A + encoded_B + encoded_C

                #decoding and we are still in generator
                decoded0 = self.dec_0(hidden_features)
                fake_0 = decoded0

                # HERE WE HSOULD COMPUTE GENERATOR LOSS functions and also loss from disc where the gen should fool the disc

                # Generator Loss function
                #fake B and all those are 1,24,128 ..

                # this is equivalant to encoded a b c to generate   fake_0~fake_B
                #fake_B = self.generator_A2B(real_A)

            ### do we need a cycle here and how to apply it
               # cycle_A = self.generator_B2A(fake_B)

                #fake_A is the generated voice from encoder0 and we should apply the radial loss here
                #fake_A = self.generator_B2A(real_B)


                # should apply the radial on this with hidden features output
                encoded0 = self.enc_0(real_0)

                decoded1 = self.decoder_List[0](encoded0)
                decoded2 = self.decoder_List[1](encoded0)
                decoded3 = self.decoder_List[2](encoded0)

            ### this should be also checked if we will use it or not
                #cycle_B = self.generator_A2B(fake_A)


            ### i think no need for identity right now
                #identity_A = self.generator_B2A(real_A)
                #identity_B = self.generator_A2B(real_B)

                d_fake_0 = self.discriminator_0 (fake_0)

                d_fake_A = self.discriminator_List[0](decoded1)
                d_fake_B = self.discriminator_List[1](decoded2)
                d_fake_C = self.discriminator_List[2](decoded3)

               # d_fake_A = self.discriminator_A(fake_A)
               # d_fake_B = self.discriminator_B(fake_B)


                # Generator Cycle loss .. check this later

            ### if it wll be applied should be real 1 with cycle 1 ,,, real2 cycle2 ... real3 cycle3 etc ...
               # cycleLoss = torch.mean(
               #     torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))
                
                # Generator Identity Loss ... not sure if we will use it
              #  identiyLoss = torch.mean(
               #     torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # Total Generator Loss
                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                    cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
                self.generator_loss_store.append(generator_loss.item())

                # Backprop for Generator
                self.reset_grad()
                generator_loss.backward()

                # if num_iterations > self.start_decay:  # Linearly decay learning rate
                #     self.adjust_lr_rate(
                #         self.generator_optimizer, name='generator')

                self.generator_optimizer.step()

                # Discriminator Loss Function

                # Discriminator Feed Forward
                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)

                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A)

                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B)

                # Loss Functions
                d_loss_A_real` = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # Final Loss for discriminator
                d_loss = (d_loss_A + d_loss_B) / 2.0
                self.discriminator_loss_store.append(d_loss.item())

                # Backprop for Discriminator
                self.reset_grad()
                d_loss.backward()

                # if num_iterations > self.start_decay:  # Linearly decay learning rate
                #     self.adjust_lr_rate(
                #         self.discriminator_optimizer, name='discriminator')

                self.discriminator_optimizer.step()
                if num_iterations % 50 == 0:
                    store_to_file = "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                        num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B, generator_loss_B2A, identiyLoss, cycleLoss, d_loss_A, d_loss_B)
                    print("Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                        num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B, generator_loss_B2A, identiyLoss, cycleLoss, d_loss_A, d_loss_B))
                    self.store_to_file(store_to_file)
            end_time = time.time()
            store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
            self.store_to_file(store_to_file)
            print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch))

            if epoch % 100 == 0 and epoch != 0:
                # Save the Entire model
                print("Saving model Checkpoint  ......")
                store_to_file = "Saving model Checkpoint  ......"
                self.store_to_file(
                        )
                self.saveModelCheckPoint(epoch, '{}'.format(
                    self.modelCheckpoint + '_CycleGAN_CheckPoint'))
                print("Model Saved!")

            if epoch % 100 == 0 and epoch != 0:
                # Validation Set
                validation_start_time = time.time()
                self.validation_for_A_dir()
                self.validation_for_B_dir()
                validation_end_time = time.time()
                store_to_file = "Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time)
                self.store_to_file(store_to_file)
                print("Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time))

    def validation_for_A_dir(self):
        num_mcep = 24
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        print("Generating Validation Data B from A...")
        for file in os.listdir(validation_A_dir):
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            librosa.output.write_wav(path=os.path.join(output_A_dir, os.path.basename(file)),
                                     y=wav_transformed,
                                     sr=sampling_rate)

    def validation_for_B_dir(self):
        num_mcep = 24
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_B_dir = self.validation_B_dir
        output_B_dir = self.output_B_dir

        print("Generating Validation Data A from B...")
        for file in os.listdir(validation_B_dir):
            filePath = os.path.join(validation_B_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_B,
                                                       std_log_src=self.log_f0s_std_B,
                                                       mean_log_target=self.log_f0s_mean_A,
                                                       std_log_target=self.log_f0s_std_A)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_B_mean) / self.coded_sps_B_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_B2A(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.coded_sps_A_std + self.coded_sps_A_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            librosa.output.write_wav(path=os.path.join(output_B_dir, os.path.basename(file)),
                                     y=wav_transformed,
                                     sr=sampling_rate)

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA': self.discriminator_A.state_dict(),
            'model_discriminatorB': self.discriminator_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder_List = []
    encoder_List.append(Encoder("first").to(device))
    encoder_List.append(Encoder("sec").to(device))

    encoder_List.append(Decoder("dec1").to(device))
    encoder_List.append(Decoder("dec2").to(device))


    for enc in  encoder_List:
        print (enc.label)


if __name__ == '__main2__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset")

    logf0s_normalization_default = '../cache_RCGan/logf0s_normalization.npz'
    mcep_normalization_default = '../cache_RCGan/mcep_normalization.npz'
    coded_sps_A_norm = '../cache_RCGan/coded_sps_s_norm0.pickle'
    coded_sps_B_norm = '../cache_RCGan/coded_sps_s_norm1.pickle'
    coded_sps_C_norm = '../cache_RCGan/coded_sps_s_norm2.pickle'
    coded_sps_D_norm = '../cache_RCGan/coded_sps_s_norm3.pickle'

    coded_sps_S_norm =[]
    coded_sps_S_norm.append(coded_sps_A_norm)
    coded_sps_S_norm.append(coded_sps_B_norm)
    coded_sps_S_norm.append(coded_sps_C_norm)
    coded_sps_S_norm.append(coded_sps_D_norm)


    model_checkpoint = '../cache_RCGan/model_checkpoint/'
    resume_training_at = '../cache_RCGan/model_checkpoint/_CycleGAN_CheckPoint'
    resume_training_at = None

    #validation_A_dir_default = '../data/vcc2016_training/evaluation_rcgan/SF2/'
    #output_A_dir_default = '../data/vcc2016_training/converted_sound_RCGAN/SF2'


    # should be now SF2 SM2 TM3 TF1
    validation_S_dir_default = '../data/vcc2016_training/evaluation_RCGAN/'
    output_S_dir_default = '../data/vcc2016_training/converted_sound_RCGAN/'



    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_S_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_S_norm)
    parser.add_argument('--model_checkpoint', type=str,
                        help="location where you want to save the model", default=model_checkpoint)
    parser.add_argument('--resume_training_at', type=str,
                        help="Location of the pre-trained model to resume training",
                        default=resume_training_at)
    parser.add_argument('--validation_S_dir', type=str,
                        help="validation set for sound source A", default=validation_S_dir_default)
    parser.add_argument('--output_S_dir', type=str,
                        help="output for converted Sound Source A", default=output_S_dir_default)


    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    #coded_sps_A_norm = argv.coded_sps_A_norm
    coded_sps_S_norm = argv.coded_sps_S_norm
    model_checkpoint = argv.model_checkpoint
    resume_training_at = argv.resume_training_at

    validation_S_dir = argv.validation_S_dir
    output_S_dir = argv.output_S_dir
    #validation_B_dir = argv.validation_B_dir
    #output_B_dir = argv.output_B_dir

    # Check whether following cached files exists
    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print(
            "Cached files do not exist, please run the program preprocess_training.py first")

    # this should take dec0 and enc0 parameters .. then the other added values ...
    cycleGAN = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                coded_sps_S_norm=coded_sps_S_norm,
                                model_checkpoint=model_checkpoint,
                                validation_S_dir=validation_S_dir,
                                output_A_dir=output_S_dir,
                                restart_training_at=resume_training_at)
    cycleGAN.train()
