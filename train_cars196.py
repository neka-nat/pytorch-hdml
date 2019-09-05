import argparse
import torch
import visdom
from hdml import train
from hdml import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hdml with triplet loss.')
    parser.add_argument('-b', '--batch_size', type=int, default=120, help="Batch size.")
    parser.add_argument('-s', '--image_size', type=int, default=227, help="The size of input images.")
    parser.add_argument('-l', '--lr_init', type=float, default=7e-5, help="Initial learning rate.")
    parser.add_argument('-m', '--max_steps', type=int, default=80000, help="The maximum step number.")
    parser.add_argument('-c', '--n_class', type=int, default=99, help="Number of classes.")
    parser.add_argument('-n', '--no_hdml', action='store_true', default=False, help='No use hdml.')
    parser.add_argument('-p', '--pretrained', action='store_true', default=False, help='Use pretrained weight.')
    parser.add_argument('-v', '--visdomserver', type=str, default='localhost', help="Visdom's server name.")
    args = parser.parse_args()
    streams = dataset.get_streams('data/CARS196/cars196.hdf5', args.batch_size, 'cars196', 'triplet', crop_size=args.image_size)
    viz = visdom.Visdom(server='http://' + args.visdomserver, log_to_filename='visdom.log')
    assert viz.check_connection(timeout_seconds=3), 'No connection could be formed quickly'

    if args.no_hdml:
        train.train_triplet(streams, viz, args.max_steps, args.n_class, args.lr_init, args.pretrained,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        train.train_hdml_triplet(streams, viz, args.max_steps, args.n_class, args.lr_init, args.pretrained,
                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))