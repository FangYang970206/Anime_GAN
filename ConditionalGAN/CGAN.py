import torch,os,scipy.misc,random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_Anime,test_Anime



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

class Generate(nn.Module):
    def __init__(self, z_dim, y_dim, image_height, image_width):
        super(Generate, self).__init__()
        self.conv_trans = nn.Sequential(
                nn.Linear(z_dim+y_dim, (image_height//16)*(image_width//16)*384),
                nn.BatchNorm1d((image_height//16)*(image_width//16)*384, 
                                eps=1e-5, momentum=0.9, affine=True),
                Reshape(-1, 384, image_height//16, image_width//16),
                nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()
        )    
            
    def forward(self, z, y):
        z = torch.cat((z,y), dim=-1)
        z = self.conv_trans(z)
        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential( 
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 384, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(407, 384, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Linear(4*4*384, 1)
            
    def forward(self, x, y):
        x = self.conv(x)
        y = torch.unsqueeze(y, 2)
        y = torch.unsqueeze(y, 3)
        y = y.expand(y.size()[0], y.size()[1], x.size()[2], x.size()[3])
        x = torch.cat((x,y), dim=1)
        x = self.conv1(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = x.squeeze()
        x = F.sigmoid(x)       
        return x

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0,0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.01)
        m.bias.data.fill_(0)

class CGAN(object):

    def __init__(self, dataset_path, save_path, epochs, batchsize, z_dim, device, mode):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batchsize
        self.mode = mode
        self.image_height = 64
        self.image_width = 64
        self.learning_rate = 0.0001
        self.z_dim = z_dim
        self.y_dim = 23
        self.iters_d = 2
        self.iters_g = 1
        self.device = device
        self.criterion = nn.BCELoss().to(device)
        if mode == "train":
            self.X, self.Y = load_Anime(self.dataset_path)
            self.batch_nums = len(self.X)//self.batch_size
        
    def train(self):
        generate = Generate(self.z_dim, self.y_dim, self.image_height, self.image_width).to(self.device)
        discriminator = Discriminator().to(self.device)
        generate.apply(weight_init)
        discriminator.apply(weight_init)
        optimizer_G = Adam(generate.parameters(), lr=self.learning_rate)
        optimizer_D = Adam(discriminator.parameters(), lr=self.learning_rate)
        step = 0
        for epoch in range(self.epochs):
            print("Main epoch:{}".format(epoch))
            for i in range(self.batch_nums):
                step += 1
                batch_images = torch.from_numpy(np.asarray(self.X[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)).to(self.device)
                batch_labels = torch.from_numpy(np.asarray(self.Y[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)).to(self.device)
                batch_images_wrong = torch.from_numpy(np.asarray(self.X[random.sample(range(len(self.X)), len(batch_images))]).astype(np.float32)).to(self.device)
                batch_labels_wrong = torch.from_numpy(np.asarray(self.Y[random.sample(range(len(self.Y)), len(batch_images))]).astype(np.float32)).to(self.device)
                batch_z = torch.from_numpy(np.random.normal(0, np.exp(-1 / np.pi), [self.batch_size, self.z_dim]).astype(np.float32)).to(self.device)
                # discriminator twice, generate once
                for _ in range(self.iters_d):
                    optimizer_D.zero_grad()
                    d_loss_real = self.criterion(discriminator(batch_images, batch_labels), torch.ones(self.batch_size).to(self.device))
                    d_loss_fake = (self.criterion(discriminator(batch_images, batch_labels_wrong), torch.zeros(self.batch_size).to(self.device)) \
                                  + self.criterion(discriminator(batch_images_wrong, batch_labels), torch.zeros(self.batch_size).to(self.device)) \
                                  + self.criterion(discriminator(generate(batch_z, batch_labels), batch_labels), torch.zeros(self.batch_size).to(self.device)))/3
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    optimizer_D.step()
                
                for _ in range(self.iters_g):
                    optimizer_G.zero_grad()
                    g_loss = self.criterion(discriminator(generate(batch_z, batch_labels), batch_labels), torch.ones(self.batch_size).to(self.device))
                    g_loss.backward()
                    optimizer_G.step()
                
                print("epoch:{}, step:{}, d_loss:{}, g_loss:{}".format(epoch, step, d_loss.item(), g_loss.item()))
                #show result and save model 
                if (step)%5000 == 0:
                    z, y = test_Anime()
                    image = generate(torch.from_numpy(z).float().to(self.device),torch.from_numpy(y).float().to(self.device)).to("cpu").detach().numpy()
                    display_grid = np.zeros((5*64,5*64,3))
                    for j in range(5):
                        for k in range(5):
                            display_grid[j*64:(j+1)*64,k*64:(k+1)*64,:] = image[k+5*j].transpose(1, 2, 0)
                    img_save_path = os.path.join(self.save_path,"training_img/{}.png".format(step))
                    scipy.misc.imsave(img_save_path, display_grid)
                    torch.save(generate, os.path.join(self.save_path, "generate.t7"))
                    torch.save(discriminator, os.path.join(self.save_path, "discriminator.t7"))

    def infer(self):
        z, y = test_Anime()
        generate = torch.load(os.path.join(self.save_path, "generate.t7")).to(self.device)
        image = generate(torch.from_numpy(z).float().to(self.device),torch.from_numpy(y).float().to(self.device)).to("cpu").detach().numpy()
        display_grid = np.zeros((5*64,5*64,3))
        for j in range(5):
            for k in range(5):
                display_grid[j*64:(j+1)*64,k*64:(k+1)*64,:] = image[k+5*j].transpose(1, 2, 0)
        img_save_path = os.path.join(self.save_path,"testing_img/test.png")
        scipy.misc.imsave(img_save_path, display_grid)
        print("infer ended, look the result in the save/testing_img/")
                    