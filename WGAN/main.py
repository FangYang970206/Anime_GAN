import torch 
import torch.nn as nn 
from torch.optim import RMSprop
from torchvision.utils import make_grid
from model import Generate,Discriminator,weight_init
from AnimeDataset import AnimeDataset 
import numpy as np
import scipy.misc
import os, argparse
from tqdm import tqdm
from utils import creat_gif

def main():

    parse = argparse.ArgumentParser()

    parse.add_argument("--lr", type=float, default=0.00005, 
                        help="learning rate of generate and discriminator")
    parse.add_argument("--clamp", type=float, default=0.01, 
                        help="clamp discriminator parameters")
    parse.add_argument("--batch_size", type=int, default=10,
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

    if not os.path.exists("saved"):
        os.mkdir("saved")
    if not os.path.exists("saved/img"):
        os.mkdir("saved/img")

    if os.path.exists("faces"):
        pass
    else:
        print("Don't find the dataset directory, please copy the link in website ,download and extract faces.tar.gz .\n \
        https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I \n ")
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
    dataload = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.loaders)

    optimizer_G = RMSprop(generate.parameters(), lr=args.lr)
    optimizer_D = RMSprop(discriminator.parameters(), lr=args.lr)

    fixed_noise = torch.randn(64, 100, 1, 1).to(device)
    step = 0
    for epoch in range(args.epochs):

        print("Main epoch{}:".format(epoch))
        progress = tqdm(total=len(dataload.dataset))
        
        for i, inp in enumerate(dataload):
            step += 1
            # train discriminator   
            real_data = inp.float().to(device)
            noise = torch.randn(inp.size()[0], 100, 1, 1).to(device)
            fake_data = generate(noise)
            optimizer_D.zero_grad()
            real_output = torch.mean(discriminator(real_data).squeeze())
            fake_output = torch.mean(discriminator(fake_data).squeeze())
            output = (real_output - fake_output)* -1
            output.backward()
            optimizer_D.step()
            
            for param in discriminator.parameters():
                param.data.clamp_(-args.clamp, args.clamp)

            #train generate
            if step%5 == 0:
                optimizer_G.zero_grad()
                fake_data = generate(noise)
                fake_output = -torch.mean(discriminator(fake_data).squeeze())
                fake_output.backward()
                optimizer_G.step()
            
            progress.update(dataload.batch_size)

        if epoch % 20 == 0:

            torch.save(generate, os.path.join(os.getcwd(), "saved/generate.t7"))
            torch.save(discriminator, os.path.join(os.getcwd(), "saved/discriminator.t7"))

            img = generate(fixed_noise).to("cpu").detach().numpy()

            display_grid = np.zeros((8*96,8*96,3))
            
            for j in range(int(64/8)):
                for k in range(int(64/8)):
                    display_grid[j*96:(j+1)*96,k*96:(k+1)*96,:] = (img[k+8*j].transpose(1, 2, 0)+1)/2

            img_save_path = os.path.join(os.getcwd(),"saved/img/{}.png".format(epoch))
            scipy.misc.imsave(img_save_path, display_grid)

    creat_gif("evolution.gif", os.path.join(os.getcwd(),"saved/img"))


                
if __name__ == "__main__":
    main()
    