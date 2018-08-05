import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.utils import make_grid
from model import Generate,Discriminator,weight_init
from AnimeDataset import AnimeDataset 
import numpy as np
import scipy.misc
import os, argparse
from tqdm import tqdm
from utils import creat_gif, visualize_loss

def main():

    parse = argparse.ArgumentParser()

    parse.add_argument("--lr", type=float, default=0.0002, 
                        help="learning rate of generate and discriminator")
    parse.add_argument("--beta1", type=float, default=0.5,
                        help="adam optimizer parameter")
    parse.add_argument("--batch_size", type=int, default=64,
                        help="number of dataset in every train or test iteration")
    parse.add_argument("--dataset", type=str, default="faces",
                        help="base path for dataset")
    parse.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parse.add_argument("--loaders", type=int, default=4,
                        help="number of parallel data loading processing")
    parse.add_argument("--size_per_dataset", type=int, default=30000,
                        help="number of training data")

    args = parse.parse_args()

    if os.path.exists("saved"):
        pass
    else:
        os.mkdir("saved")
        os.mkdir("saved/img")

    if os.path.exists("faces"):
        pass
    else:
        print("Don't find the dataset directory, please copy the link in website and download faces.tar.gz .\n \
        https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I  \n \
        At last, rename the folder to anime")
        exit()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    generate = Generate().to(device)
    discriminator = Discriminator().to(device)

    generate.apply(weight_init)
    discriminator.apply(weight_init)

    dataset = AnimeDataset(os.getcwd(), args.dataset, args.size_per_dataset)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    criterion = nn.BCELoss().to(device)

    optimizer_G = Adam(generate.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    for epoch in range(args.epochs):

        print("Main epoch{}:".format(epoch))
        progress = tqdm(total=len(dataload.dataset))
        loss_d, loss_g = 0, 0

        for i, inp in enumerate(dataload):
            # train discriminator   
            real_data = inp.float().to(device)
            real_label = torch.ones(inp.size()[0]).to(device)
            noise_D = torch.rand(inp.size()[0], 100, 1, 1).to(device)
            fake_data_D = generate(noise_D)
            fake_label_D = torch.zeros(fake_data_D.size()[0]).to(device)
            optimizer_D.zero_grad()
            real_output = discriminator(real_data)
            real_loss = criterion(real_output.squeeze(), real_label)
            real_loss.backward()
            fake_output_D = discriminator(fake_data_D)
            fake_loss = criterion(fake_output_D.squeeze(), fake_label_D)
            fake_loss.backward()
            loss_D = real_loss + fake_loss
            optimizer_D.step()

            #train generate
            noise_G = torch.rand(inp.size()[0]*2, 100, 1, 1).to(device)
            optimizer_G.zero_grad()
            fake_data_G = generate(noise_G)
            fake_label_G = torch.ones(fake_data_G.size()[0]).to(device)
            fake_output_G = discriminator(fake_data_G)
            loss_G = criterion(fake_output_G.squeeze(), fake_label_G)
            loss_G.backward()
            optimizer_G.step()

            progress.update(dataload.batch_size)
            progress.set_description("D:{}, G:{}".format(loss_D.item(), loss_G.item()))

            loss_g += loss_G.item()
            loss_d += loss_D.item()
        
        loss_g /= (i+1)
        loss_d /= (i+1)

        with open("generate_loss.txt", 'a+') as f:
            f.write("loss_G:{} \n".format(loss_G.item()))

        with open("discriminator_loss.txt", 'a+') as f:
            f.write("loss_D:{} \n".format(loss_D.item()))

        if epoch % 20 == 0:

            torch.save(generate, os.path.join(os.getcwd(), "saved/generate.t7"))
            torch.save(discriminator, os.path.join(os.getcwd(), "saved/discriminator.t7"))

            noise = torch.rand(64, 100, 1, 1).to(device)
            img = generate(noise).to("cpu").detach().numpy()

            display_grid = np.zeros((8*96,8*96,3))
            print(dataload.batch_size)
            for j in range(int(64/8)):
                for k in range(int(64/8)):
                    display_grid[j*96:(j+1)*96,k*96:(k+1)*96,:] = (img[k+8*j].transpose(1, 2, 0)+1)/2

            display_grid = np.clip(display_grid, 0, 1)
            img_save_path = os.path.join(os.getcwd(),"saved/img/{}.png".format(epoch))
            scipy.misc.imsave(img_save_path, display_grid)

    creat_gif("evolution.gif", os.path.join(os.getcwd(),"saved/img"))

    visualize_loss("generate_loss.txt", "discriminator_loss.txt")

                


if __name__ == "__main__":
    main()
    