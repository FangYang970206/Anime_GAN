import argparse
from CGAN import CGAN
from utils import check_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--dataset_path', type=str, default='./images/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--save_path', type=str, default='./save/',
                        help='Directory name to save the generated images')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='train on GPU or CPU')
    parser.add_argument('--save_training_img_path', type=str, default='./save/training_img/',
                        help='Directory name to save the training images')
    parser.add_argument('--save_testing_img_path', type=str, default='./save/testing_img/',
                        help='Directory name to save the training images')
    return parser.parse_args()


def main():
    args = parse_args()
    check_folder(args.dataset_path)
    check_folder(args.save_path)
    gan = CGAN(args.dataset_path,
               args.save_path,
               args.epochs,
               args.batch_size,
               args.z_dim,
               args.device,
               args.mode)
    if args.mode == "train":
        check_folder(args.save_training_img_path)
        gan.train()
    else:
        check_folder(args.save_testing_img_path)
        gan.infer()    

if __name__ == "__main__":
    main()