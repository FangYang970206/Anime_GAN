# most code from https://github.com/JasonYao81000/MLDS2018SPRING/blob/master/hw3/hw3_2/
import numpy as np
import cv2
import os

def test_Anime():
    np.random.seed(999)
    z = np.random.normal(0, np.exp(-1 / np.pi), [25, 62])
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    tag_txt = open("test.txt", 'r').readlines()
    labels = []
    for line in tag_txt:
        label = np.zeros(len(tag_dict))

        for i in range(len(tag_dict)):
            if tag_dict[i] in line:
                label[i] = 1
        labels.append(label)

    for i in range(len(tag_txt)):
        for j in range(4):
            labels.insert(5*i+j, labels[5*i])
    
    return z, np.array(labels)


def load_Anime(dataset_filepath):
    tag_csv_filename = dataset_filepath.replace('images/', 'tags.csv')
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
            'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
            'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
            'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

    tag_csv = open(tag_csv_filename, 'r').readlines()

    id_label = []
    for line in tag_csv:
        id, tags = line.split(',')
        label = np.zeros(len(tag_dict))
        
        for i in range(len(tag_dict)):
            if tag_dict[i] in tags:
                label[i] = 1
        
        # Keep images with hair or eyes.
        if np.sum(label) == 2 or np.sum(label) == 1:
            id_label.append((id, label))


    # Load file name of images.
    image_file_list = []
    for image_id, _ in id_label:
        image_file_list.append(image_id + '.jpg')

    # Resize image to 64x64.
    image_height = 64
    image_width = 64
    image_channel = 3

    # Allocate memory space of images and labels.
    images = np.zeros((len(image_file_list), image_channel, image_width, image_height))
    labels = np.zeros((len(image_file_list), len(tag_dict)))
    print ('images.shape: ', images.shape)
    print ('labels.shape: ', labels.shape)

    print ('Loading images to numpy array...')
    data_dir = dataset_filepath
    for index, filename in enumerate(image_file_list):
        images[index] = cv2.cvtColor(
            cv2.resize(
                cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_COLOR), 
                (image_width, image_height)), 
                cv2.COLOR_BGR2RGB).transpose(2,0,1)
        labels[index] = id_label[index][1]
    
    print ('Random shuffling images and labels...')
    np.random.seed(9487)
    indice = np.array(range(len(image_file_list)))
    np.random.shuffle(indice)
    images = images[indice]
    labels = labels[indice]

    print ('[Tip 1] Normalize the images between -1 and 1.')
    # Tip 1. Normalize the inputs
    #   Normalize the images between -1 and 1.
    #   Tanh as the last layer of the generator output.
    return (images / 127.5) - 1, labels

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
