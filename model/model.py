import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c): #input channels and output channels
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1) #1st convolutional layer (3x3)
        self.bn1 = nn.BatchNorm2d(out_c) #batch normalisation

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1) #2nd layer
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU() #relu

    def forward(self, inputs): #where all operations take place
        x = self.conv1(inputs) #first convolution layer
        x = self.bn1(x) #batch normalisation
        x = self.relu(x) #relu

        x = self.conv2(x) #input is output of first layer
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module): #convolultional layer (above) followed by max pooling
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2)) #(2x2) reduces size 128 -> 64 for example (half as 2x2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0) # (2x2) transpose convolution. stride of 2 to upsample
        self.conv = conv_block(out_c+out_c, out_c) #normal conv block

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1) #brings info from encoder to decoder to help feature map?
        x = self.conv(x) #normal conv block
        return x

class build_unet(nn.Module): #put all layers and blocks together
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64) # input = 3 channels, output = 64 channels [2, 64, 512, 512]
        self.e2 = encoder_block(64, 128) # [2, 128, 256, 256]
        self.e3 = encoder_block(128, 256) # [2, 256, 128, 128]
        self.e4 = encoder_block(256, 512) # [2, 512, 64, 64]

        """ Bottleneck """
        self.b = conv_block(512, 1024) #just conv block nothing else - bottleneck block or bridge [2, 1024, 32, 32]

        """ Decoder """
        self.d1 = decoder_block(1024, 512) #[]
        self.d2 = decoder_block(512, 256) #[]
        self.d3 = decoder_block(256, 128) #[]
        self.d4 = decoder_block(128, 64) #[]

        """ Classifier """ # to generate segmentation map
        #output convolutional layer
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0) #output channel = 1 (as its binary seg) - generates seg mask of 512 by 512

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512)) # 2 = batch size and 3 = channels, 512 = height, 512 = width
    f = build_unet()
    y = f(x)
    print(y.shape)
