import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


class Parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse(self) -> Namespace:
        parser_ = argparse.ArgumentParser()
        parser_.add_argument('-nd', '--number_drives', default=1, type=int, help='number of drives to use in dataset')
        parser_.add_argument('-ndtrain', '--starting-drive-train', default=400, type=int, help='starting number of drive to use for training dataset')
        parser_.add_argument('-ndtest', '--starting-drive-test', default=500, type=int, help="starting number of drive to use for test dataset")
        parser_.add_argument('-seq', '--sequence-length', default=5, type=int, help='Number time series steps to look back')
        parser_.add_argument('-nnsize', '--neuralnetwork-size', default=64, type=int, help='Neural network size')
        parser_.add_argument('-nnlayers', '--neuralnetwork-layers', default=2, type=int, help='Neural network number of layers')
        parser_.add_argument('-lstm', '--lstm_enable', default=1, type=int, help='Train LSTM or not')
        parser_.add_argument('-cnn', '--cnn_enable', default=0, type=int, help='Train CNN or not')
        parser_.add_argument('-balanced', '--bdataset', default=1, type=int, help='Generate balanced dataset or not')
        parser_.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size')
        parser_.add_argument('-epoch', '--epoch_number', default=1000, type=int, help='number of epochs')
        parser_.add_argument('-lr', '--learn_rate', default=0.000001, type=float, help='learning rate')
        parser_.add_argument('-wd', '--weight_decay', default=0.0000001, type=float, help='weight decay')
        parser_.add_argument('-lff', '--load_from_files', default=1, type=int, help='if to preprocess or not the data')
        parser_.add_argument('-mso', '--max_switch_over', default=0, type=int, help='Generate data set with maximum switchover per drive')
        args = parser_.parse_args()
        return args
