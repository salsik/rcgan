



###############################################

## edited one according to this repo


#  https://github.com/aitorzip/PyTorch-CycleGAN




###################################################







import torch.nn as nn
import torch
import numpy as np


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)


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


class Generator_old(nn.Module):
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


class Discriminator_old(nn.Module):
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



###################################


#  https://github.com/aitorzip/PyTorch-CycleGAN

##################################


class ResidualBlock11(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator11(nn.Module):
    def __init__(self, input_nc=24, output_nc=24, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator11(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)



###################################


#https://github.com/eriklindernoren/PyTorch-GAN/tree/a163b82beff3d01688d8315a3fd39080400e7c01/implementations/cyclegan


##################################



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            #nn.ReflectionPad2d(1),
            nn.Conv1d(in_features, in_features, 3 ,stride=1, padding=1),
            nn.InstanceNorm1d(in_features),
            nn.ReLU(inplace=False),
            #nn.ReflectionPad2d(1),
            nn.Conv1d(in_features, in_features, 3,stride=1, padding=1),
            nn.InstanceNorm1d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)



class Generator(nn.Module):
    def __init__(self, input_shape= [24], num_residual_blocks=9):
        super(Generator, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            #nn.ReflectionPad2d(channels),
            nn.Conv1d(channels, out_features, 7),
            nn.InstanceNorm1d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv1d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm1d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv1d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm1d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [#nn.ReflectionPad2d(channels),
                  nn.Conv1d(out_features, channels, 3,stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape=24):
        super(Discriminator, self).__init__()

        channels = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        #self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            #nn.ZeroPad1d((1, 0, 1, 0)),
            nn.Conv1d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)



if __name__ == '__main__':
    # Generator Dimensionality Testing
    #input = torch.randn(10, 24, 1100)  # (N, C_in, Width) For Conv1d
    np.random.seed(0)
    print(np.random.randn(10))
    input = np.random.randn(158, 24,150,50)
    input = torch.from_numpy(input).float()
    # print(input)


    input = torch.randn(1, 24, 128)
    generator = Generator()
    output = generator(input)

    print("Input shape Generator", input.shape)
    print("Output shape Generator", output.shape)

    # Discriminator Dimensionality Testing
    # input = torch.randn(32, 1, 24, 128)  # (N, C_in, height, width) For Conv2d
    discriminator = Discriminator()
    output = discriminator(output)
    print("Output shape Discriminator", output.shape)
