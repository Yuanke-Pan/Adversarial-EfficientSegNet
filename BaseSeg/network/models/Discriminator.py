# Wait to modify
from base64 import encode
from json import encoder
import torch
import torch.nn as nn

from BaseSeg.network.blocks.basic_unit import _ConvINReLU3D
from BaseSeg.network.blocks.context_block import AnisotropicMaxPooling, AnisotropicAvgPooling
from BaseSeg.network.blocks.process_block import InputLayer, OutputLayer
from BaseSeg.network.blocks.residual_block import ResBaseConvBlock, AnisotropicConvBlock


class Discriminator(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        # ContextUNet parameter.
        num_class = cfg['NUM_CLASSES']
        num_channel = cfg['NUM_CHANNELS']
        num_blocks = cfg['NUM_BLOCKS']
        decoder_num_block = cfg['DECODER_NUM_BLOCK']
        self.num_depth = cfg['NUM_DEPTH']
        self.is_preprocess = cfg['IS_PREPROCESS']
        self.is_postprocess = cfg['IS_POSTPROCESS']
        self.auxiliary_task = cfg['AUXILIARY_TASK']
        self.auxiliary_class = cfg['AUXILIARY_CLASS']
        self.is_dynamic_empty_cache = cfg['IS_DYNAMIC_EMPTY_CACHE']

        if cfg['ENCODER_CONV_BLOCK'] == 'AnisotropicConvBlock':
            encoder_conv_block = AnisotropicConvBlock
        else:
            encoder_conv_block = ResBaseConvBlock

        if cfg['CONTEXT_BLOCK'] == 'AnisotropicMaxPooling':
            context_block = AnisotropicMaxPooling
        elif cfg['CONTEXT_BLOCK'] == 'AnisotropicAvgPooling':
            context_block = AnisotropicAvgPooling
        else:
            context_block = None
        # -----------------------------------------
        # test zone
        context_block = None
        ndf = 64
        #------------------------------------------
        self.input = InputLayer(input_size=cfg['INPUT_SIZE'], clip_window=cfg['WINDOW_LEVEL'])
        self.output = OutputLayer()
        if cfg['INPUT_SIZE'][0] == 160:
            self.pool = nn.MaxPool3d(kernel_size=10)
        elif cfg['INPUT_SIZE'][0] == 192:
            self.pool = nn.MaxPool3d(kernel_size=12)
        self.conv0_0 = encoder_conv_block(num_class, ndf, stride = 2)
        self.conv0_1 = encoder_conv_block(num_class, ndf, stride = 2)
        # There are two feature extractor for each input.
        self.conv1 = encoder_conv_block(1, ndf, stride = 2)

        self.conv2 = encoder_conv_block(ndf, ndf * 2, stride = 2)
        self.conv3 = encoder_conv_block(ndf * 2, ndf * 4, stride = 2)
        self.conv4 = encoder_conv_block(ndf * 4, ndf * 8, stride = 2)
        # 160*160*160 / 16 /16 / 16
        self.classifier = nn.Linear(ndf * 8 * 3, 2)

        self.softmax = nn.Softmax()

        if context_block is not None:
            context_kernel_size = [i//16 for i in cfg['INPUT_SIZE']]
            self.context_block = context_block(num_channel[4], num_channel[4], kernel_size=context_kernel_size,
                                               is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        else:
            self.context_block = nn.Sequential()



        self._initialize_weights()
        # self.final.bias.data.fill_(-2.19)

    def _mask_layer(self, block, in_channels, out_channels, num_block, stride):
        layers = []
        layers.append(block(in_channels, out_channels, p=0.2, stride=stride, is_identify=False,
                            is_dynamic_empty_cache=self.is_dynamic_empty_cache))
        for _ in range(num_block-1):
            layers.append(block(out_channels, out_channels, p=0.2, stride=1, is_identify=True,
                                is_dynamic_empty_cache=self.is_dynamic_empty_cache))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image, target0, target1):

        batch_size = target0.shape[0]

        out_size = image.shape[2:]
        if self.is_preprocess:
            image = self.input(image)


        image = self.conv1(image)
        target0 = self.conv0_0(target0)
        target1 = self.conv0_1(target1)

        x = torch.cat((image, target0, target1), dim=0) # The dimension maybe is no difference there, when we didn't 
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        x = self.softmax(x)
        return x
