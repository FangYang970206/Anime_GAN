import os, imageio,scipy.misc
import matplotlib.pyplot as plt


def creat_gif(gif_name, img_path, duration=0.3):
    frames = []
    img_names = os.listdir(img_path)
    img_list = [os.path.join(img_path, img_name) for img_name in img_names]
    for img_name in img_list:
        frames.append(imageio.imread(img_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

def visualize_loss(generate_txt_path, discriminator_txt_path):
    
    with open(generate_txt_path, 'r') as f:
        G_list_str = f.readlines()

    with open(discriminator_txt_path, 'r') as f:
        D_list_str = f.readlines()
    
    D_list_float, G_list_float = [], []

    for D_item, G_item in zip(D_list_str, G_list_str):
        D_list_float.append(float(D_item.strip().split(':')[-1]))
        G_list_float.append(float(G_item.strip().split(':')[-1]))
    
    list_epoch = list(range(len(D_list_float)))

    full_path = os.path.join(os.getcwd(), "saved/logging.png")
    plt.figure()
    plt.plot(list_epoch, G_list_float, label="generate", color='g')
    plt.plot(list_epoch, D_list_float, label="discriminator", color='b')
    plt.legend()
    plt.title("DCGAN_Anime")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(full_path)


    
    