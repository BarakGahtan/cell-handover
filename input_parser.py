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
        parser_.add_argument('-seq', '--sequence-length', default=5, type=int, help='Number time series steps to look back')
        parser_.add_argument('-nnsize', '--neuralnetwork-size', default=64, type=int, help='Neural network size')
        parser_.add_argument('-nnlayers', '--neuralnetwork-layers', default=2, type=int, help='Neural network number of layers')
        parser_.add_argument('-balanced', '--bdataset', default=1, type=int,
                             help='if 1 then under sample the data, if 0 then use SMOTE to balance the data')
        parser_.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size')
        parser_.add_argument('-epoch', '--epoch_number', default=1000, type=int, help='number of epochs')
        parser_.add_argument('-lr', '--learn_rate', default=0.001, type=float, help='learning rate')
        parser_.add_argument('-lff', '--load_from_files', default=1, type=int, help='if to preprocess or not the data')
        parser_.add_argument('-mso', '--max_switch_over', default=0, type=int, help='Generate data set with maximum switchover per drive')
        parser_.add_argument('-mdimsi', '--max_data_imsi', default=0, type=int, help='Generate data set with most data per imsi')
        parser_.add_argument('-name', '--model_name', default="", type=str, help='name of the model')
        parser_.add_argument('-tt', '--to_train', default=0, type=int, help='If to send the model to train')
        parser_.add_argument('-l', '--label', default=1, type=int, help='label = 1 = switchover, 0 = latency, 2 = loss')
        args = parser_.parse_args()
        return args
