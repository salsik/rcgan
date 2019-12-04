import torch.nn as nn
import torch
import numpy as np
import os
import argparse
import time
import librosa
import pickle

import preprocess
from trainingDataset import trainingDataset




class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        self.generator_lr = 0.0002
        self.discriminator_lr = 0.0001

        # Learning rate decay
        self.generator_lr_decay = self.generator_lr / 200000
        self.discriminator_lr_decay = self.discriminator_lr / 200000

        # Starts learning rate decay from after this many iterations have passed
        self.start_decay = 200000


        g_params = list(self.generator_A2B.parameters()) + \
                   list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
                   list(self.discriminator_B.parameters())


        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))


    def forward(self, input):
        return input * torch.sigmoid(input)

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        # Training Begins
        # from 0 to 5000 epochs
        for epoch in range(0, 100):
            start_time_epoch = time.time()

            # Constants
            cycle_loss_lambda = 10
            identity_loss_lambda = 5

            input = np.random.randn(158, 24, 256)
            dataset_A = list(input)

            input = np.random.randn(158, 24, 256)
            dataset_B = list(input)

            # Preparing Dataset 162 voice sample as defined first
            n_samples = len(dataset_A)

            dataset = trainingDataset(datasetA=dataset_A,
                                      datasetB=dataset_B,
                                      n_frames=256)
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       drop_last=False)
            # real A is from the shape 1,24,128 and real B as well
            for i, (real_A, real_B) in enumerate(train_loader):

                print("iteration "+str(i) + "\n")

                num_iterations = (
                                         n_samples // 1) * epoch + i
                # print("iteration no: ", num_iterations, epoch)

                if num_iterations > 10000:
                    identity_loss_lambda = 0
                if num_iterations > 10000:
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')

                real_A = real_A.to(self.device).float()
                real_B = real_B.to(self.device).float()

                # Generator Loss function
                # fake B and all those are 1,24,128 ... try to change 128 which is frame size to another thing and let's see the changes
                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)


                # Generator Cycle loss
                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                # Generator Identity Loss
                identiyLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # Total Generator Loss
                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                                 cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
             #   self.generator_loss_store.append(generator_loss.item())

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
                d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # Final Loss for discriminator
                d_loss = (d_loss_A + d_loss_B) / 2.0
            #    self.discriminator_loss_store.append(d_loss.item())

                # Backprop for Discriminator
                self.reset_grad()
                d_loss.backward()

                # if num_iterations > self.start_decay:  # Linearly decay learning rate
                #     self.adjust_lr_rate(
                #         self.discriminator_optimizer, name='discriminator')

                self.discriminator_optimizer.step()
                if num_iterations % 50 == 0:
                    store_to_file = "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                        num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B, generator_loss_B2A,
                        identiyLoss, cycleLoss, d_loss_A, d_loss_B)
                    print(
                        "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                            num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B,
                            generator_loss_B2A,
                            identiyLoss, cycleLoss, d_loss_A, d_loss_B))
                  #  self.store_to_file(store_to_file)
            end_time = time.time()
            store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
           # self.store_to_file(store_to_file)
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


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        # self.residualLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
        #                                              out_channels=out_channels,
        #                                              kernel_size=kernel_size,
        #                                              stride=1,
        #                                              padding=padding),
        #                                    nn.InstanceNorm1d(
        #                                        num_features=out_channels,
        #                                        affine=True),
        #                                    GLU(),
        #                                    nn.Conv1d(in_channels=out_channels,
        #                                              out_channels=in_channels,
        #                                              kernel_size=kernel_size,
        #                                              stride=1,
        #                                              padding=padding),
        #                                    nn.InstanceNorm1d(
        #                                        num_features=in_channels,
        #                                        affine=True)
        #                                    )

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm1d(num_features=out_channels,
                                                               affine=True))

    def forward(self, input):
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


class upSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(upSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm1d(num_features=out_channels // 2,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             PixelShuffle(upscale_factor=2),
                                             nn.InstanceNorm1d(num_features=out_channels // 2,
                                                               affine=True))

    def forward(self, input):
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()



        self.conv1 = nn.Conv1d(in_channels=24,
                               out_channels=128,
                               kernel_size=15,
                               stride=1,
                               padding=7)

        self.conv1_gates = nn.Conv1d(in_channels=24,
                               out_channels=128,
                               kernel_size=15,
                               stride=1,
                               padding=7)

        # Downsample Layer
        self.downSample1 = downSample_Generator(in_channels=128,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=1)

        self.downSample2 = downSample_Generator(in_channels=256,
                                                out_channels=512,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        # UpSample Layer
        self.upSample1 = upSample_Generator(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=5,
                                            stride=1,
                                            padding=2)

        self.upSample2 = upSample_Generator(in_channels=1024 // 2,
                                            out_channels=512,
                                            kernel_size=5,
                                            stride=1,
                                            padding=2)

        self.lastConvLayer = nn.Conv1d(in_channels=512 // 2,
                                       out_channels=24,
                                       kernel_size=15,
                                       stride=1,
                                       padding=7)

    def forward(self, input):
        # GLU
        conv1 = self.conv1(input) * torch.sigmoid(self.conv1_gates(input))

        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)
        residual_layer_1 = self.residualLayer1(downsample2)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)
        upSample_layer_1 = self.upSample1(residual_layer_6)
        upSample_layer_2 = self.upSample2(upSample_layer_1)
        output = self.lastConvLayer(upSample_layer_2)
        return output


class DownSample_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample_Discriminator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayerGates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding),
                                            nn.InstanceNorm2d(num_features=out_channels,
                                                              affine=True))

    def forward(self, input):
        # GLU
        return self.convLayer(input) * torch.sigmoid(self.convLayerGates(input))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Conv2d(in_channels=1,
                                    out_channels=128,
                                    kernel_size=[3, 3],
                                    stride=[1, 2])
        self.convLayer1_gates = nn.Conv2d(in_channels=1,
                                          out_channels=128,
                                          kernel_size=[3, 3],
                                          stride=[1, 2])

        # Note: Kernel Size have been modified in the PyTorch implementation
        # compared to the actual paper, as to retain dimensionality. Unlike,
        # TensorFlow, PyTorch doesn't have padding='same', hence, kernel sizes
        # were altered to retain the dimensionality after each layer

        # DownSample Layer
        self.downSample1 = DownSample_Discriminator(in_channels=128,
                                                    out_channels=256,
                                                    kernel_size=[3, 3],
                                                    stride=[2, 2],
                                                    padding=0)

        self.downSample2 = DownSample_Discriminator(in_channels=256,
                                                    out_channels=512,
                                                    kernel_size=[3, 3],
                                                    stride=[2, 2],
                                                    padding=0)

        self.downSample3 = DownSample_Discriminator(in_channels=512,
                                                    out_channels=1024,
                                                    kernel_size=[6, 3],
                                                    stride=[1, 2],
                                                    padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=1024,
                            out_features=1)

    # def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
    #     convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
    #                                         out_channels=out_channels,
    #                                         kernel_size=kernel_size,
    #                                         stride=stride,
    #                                         padding=padding),
    #                               nn.InstanceNorm2d(num_features=out_channels,
    #                                                 affine=True),
    #                               GLU())
    #     return convLayer

    def forward(self, input):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        input = input.unsqueeze(1)
        # GLU
        pad_input = nn.ZeroPad2d((1, 0, 1, 1))
        layer1 = self.convLayer1(
            pad_input(input)) * torch.sigmoid(self.convLayer1_gates(pad_input(input)))

        pad_input = nn.ZeroPad2d((1, 0, 1, 0))
        downSample1 = self.downSample1(pad_input(layer1))

        pad_input = nn.ZeroPad2d((1, 0, 1, 0))
        downSample2 = self.downSample2(pad_input(downSample1))

        pad_input = nn.ZeroPad2d((1, 0, 3, 2))
        downSample3 = self.downSample3(pad_input(downSample2))

        downSample3 = downSample3.contiguous().permute(0, 2, 3, 1).contiguous()
        # fc = torch.sigmoid(self.fc(downSample3))
        # Taking off sigmoid layer to avoid vanishing gradient problem
        fc = self.fc(downSample3)
        return fc


if __name__ == '__main1__':
    # Generator Dimensionality Testing
    input = torch.randn(10, 24, 1100)  # (N, C_in, Width) For Conv1d
    np.random.seed(0)
    print(np.random.randn(10))
    input = np.random.randn(158, 24, 128)
    input = torch.from_numpy(input).float()
    # print(input)
    generator = Generator()
    output = generator(input)
    print("Output shape Generator", output.shape)

    # Discriminator Dimensionality Testing
    # input = torch.randn(32, 1, 24, 128)  # (N, C_in, height, width) For Conv2d
    discriminator = Discriminator()
    output = discriminator(output)
    print("Output shape Discriminator", output.shape)




if __name__ == '__main3__':
    # Generator Dimensionality Testing
    input = torch.rand(11, 24, 7)  # (N, C_in, Width) For Conv1d
    np.random.seed(0)
    #print(np.random.randn(10))
    input = np.random.randn(4, 24, 16)
    input = torch.from_numpy(input).float()
    # print(input)
    generator = Generator()
    output = generator(input)
    print("Output shape Generator", output.shape)
    gen4 = Generator()
    gen5 = Generator()

    gen2 = Generator()
    out2 = gen2(input)
    out3 = gen2(input)
    out4 = gen4(input)
    out5 = gen4(input)
    # Discriminator Dimensionality Testing
    # input = torch.randn(32, 1, 24, 128)  # (N, C_in, height, width) For Conv2d
    discriminator = Discriminator()
    output_disc = discriminator(output)
    output_disc2 = discriminator(out2)
    output_disc3 = discriminator(out3)
    output_disc4 = discriminator(out4)
    output_disc5 = discriminator(out5)
    print("Output shape Discriminator", output_disc.shape)


 #train here as a regular gen and disc


if __name__ == '__main__':
    cycleGAN = GLU()
    cycleGAN.train()
    # Generator Dimensionality Testing
    input = torch.rand(11, 24, 7)  # (N, C_in, Width) For Conv1d
    np.random.seed(0)
    # print(np.random.randn(10))
    input = np.random.randn(4, 24, 16)
    input = torch.from_numpy(input).float()

    # print(input)
    generator = Generator()
    output = generator(input)
    print("Output shape Generator", output.shape)
    gen4 = Generator()
    gen5 = Generator()

    gen2 = Generator()
    out2 = gen2(input)
    out3 = gen2(input)
    out4 = gen4(input)
    out5 = gen4(input)
    # Discriminator Dimensionality Testing
    # input = torch.randn(32, 1, 24, 128)  # (N, C_in, height, width) For Conv2d
    discriminator = Discriminator()
    output_disc = discriminator(output)
    output_disc2 = discriminator(out2)
    output_disc3 = discriminator(out3)
    output_disc4 = discriminator(out4)
    output_disc5 = discriminator(out5)
    print("Output shape Discriminator", output_disc.shape)