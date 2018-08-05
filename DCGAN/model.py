import torch 
import torch.nn as nn 
import torch.functional as F

class Generate(nn.Module):
    def __init__(self, input_dim=100):
        super(Generate, self).__init__()
        channel = [512, 256, 128, 64, 3]
        kernel_size = 4
        stride = 2
        padding = 1
        self.convtrans1_block = self.__convtrans_bolck(input_dim, channel[0], 6, padding=0, stride=stride)
        self.convtrans2_block = self.__convtrans_bolck(channel[0], channel[1], kernel_size, padding, stride)
        self.convtrans3_block = self.__convtrans_bolck(channel[1], channel[2], kernel_size, padding, stride)
        self.convtrans4_block = self.__convtrans_bolck(channel[2], channel[3], kernel_size, padding, stride)
        self.convtrans5_block = self.__convtrans_bolck(channel[3], channel[4], kernel_size, padding, stride, layer="last_layer")
    
    def __convtrans_bolck(self, in_channel, out_channel, kernel_size, padding, stride, layer=None):
        if layer == "last_layer":
            convtrans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
            tanh = nn.Tanh()
            return nn.Sequential(convtrans, tanh)
        else:
            convtrans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
            batch_norm = nn.BatchNorm2d(out_channel)
            relu = nn.ReLU(True)
            return nn.Sequential(convtrans, batch_norm, relu)

    def forward(self, inp):
        x = self.convtrans1_block(inp)
        x = self.convtrans2_block(x)
        x = self.convtrans3_block(x)
        x = self.convtrans4_block(x)
        x = self.convtrans5_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        channels = [3, 64, 128, 256, 512]
        kernel_size = 4
        stride = 2
        padding = 1
        self.conv_bolck1 = self.__conv_block(channels[0], channels[1], kernel_size, stride, padding, "first_layer")
        self.conv_bolok2 = self.__conv_block(channels[1], channels[2], kernel_size, stride, padding)
        self.conv_bolok3 = self.__conv_block(channels[2], channels[3], kernel_size, stride, padding)
        self.conv_bolok4 = self.__conv_block(channels[3], channels[4], kernel_size, stride, padding)
        self.conv_bolok5 = self.__conv_block(channels[4], 1, kernel_size+1, stride, 0, "last_layer") 

    def __conv_block(self, inchannel, outchannel, kernel_size, stride, padding, layer=None):
        if layer == "first_layer":
            conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, leakrelu)
        elif layer == "last_layer":
            conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False)
            sigmoid = nn.Sigmoid()
            return nn.Sequential(conv, sigmoid)
        else:
            conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False)
            batchnorm = nn.BatchNorm2d(outchannel)
            leakrelu = nn.LeakyReLU(0.2, inplace=True)
            return nn.Sequential(conv, batchnorm, leakrelu)

    def forward(self,inp):
        x = self.conv_bolck1(inp)
        x = self.conv_bolok2(x)
        x = self.conv_bolok3(x)
        x = self.conv_bolok4(x)
        x = self.conv_bolok5(x)
        return x 


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0,0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.01)
        m.bias.data.fill_(0)



if __name__ == "__main__":
    model1 = Generate()
    x = torch.randn(10,100,1,1)
    y = model1.forward(x)
    print(y.size())
    model2 = Discriminator()
    a = torch.randn(10,3,96,96)
    b = model2.forward(a)
    print(b.size())

