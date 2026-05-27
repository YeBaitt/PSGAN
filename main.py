# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import SiameseNet

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import time
import random
from colorama import Fore, init
import json


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': 2*train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        if train_config['loader_mode'] == 0:
            train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

            self.train_dataset = train_dataset
            self.test_dataset = test_dataset


            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                shuffle=False, num_workers=0)
        elif train_config['loader_mode'] == 1:
            test_dataloader, val_dataloader = self.get_loaders(test_dataset, train_config['seed'], train_config['batch'], mode=train_config['loader_mode'], val_ratio = train_config['val_ratio'])
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'],
                                shuffle=False, num_workers=0)
            self.val_dataloader = val_dataloader
            self.test_dataloader = test_dataloader
        elif train_config['loader_mode'] == 2:
            train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

            self.train_dataset = train_dataset
            self.test_dataset = test_dataset


            self.train_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                shuffle=False, num_workers=0)
            self.val_dataloader = val_dataloader
            self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = SiameseNet(edge_index_sets, len(feature_map), train_config).to(self.device)



    def run(self):
        init()
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
            log_save_path = self.get_save_path()[1]
            with open(log_save_path, 'a') as f:
                print(f"RESTART TRAINING FROM {model_save_path}", file=f)

            self.model.load_state_dict(torch.load(model_save_path))

        else:
            model_save_path = self.get_save_path()[0]
            log_save_path = self.get_save_path()[1]
        
        with open(log_save_path, 'a') as f:
                print(train_config['comment'], file=f)
                print(json.dumps(train_config, indent=4), file=f)
                print(json.dumps(env_config, indent=4), file=f, end="\n\n")
                print(f"INITIATING TRAIN", file=f)
        start = time.time()
        train(self.model, model_save_path, log_save_path,
            config = train_config,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader, 
            feature_map=self.feature_map,
            test_dataloader=self.test_dataloader,
            test_dataset=self.test_dataset,
            train_dataset=self.train_dataset,
            dataset_name=self.env_config['dataset'])
        end = time.time()
        train_time = (end-start)/60
        print(Fore.GREEN + f"TRAINING CONSUMED {train_time:.2f} MINUTES")
        with open(log_save_path, 'a') as f:
            print(f"TRAINING CONSUMED {train_time:.2f} MINUTES", file=f)
            

        # test
        print(Fore.GREEN + f"INITIATING TEST")
        with open(log_save_path, 'a') as f:
            print(f"INITIATING TEST", file=f)
        start = time.time() 
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)  # dataloader全部数据切片的结果

        self.get_score(log_save_path, train_config, self.test_result, self.val_result)
        end = time.time()
        train_time = (end-start)/60
        print(Fore.GREEN + f"TESTING CONSUMED {train_time:.2f} MINUTES")
        with open(log_save_path, 'a') as f:
            print(f"TESTING CONSUMED {train_time:.2f} MINUTES", file=f)

    def get_testdata(self, dataset, seed, batch):
        def find_longest_zeros_segment(tensor):  
            # 确保tensor是一维的  
            if tensor.dim() != 1:  
                raise ValueError("Tensor must be 1-dimensional.")  
            # 创建一个标记tensor，其中1表示当前元素是0串的开始，0表示其他情况  
            # 逻辑是：(tensor == 0) & ((tensor[:-1] == 1) | (tensor[1:] == 1)) 的逻辑反  
            # 这里的操作是通过异或（^）来找出0串的开始位置，并且考虑到tensor的首尾  
            start_mask = (tensor[:-1] != tensor[1:]) & (tensor[:-1] == 0) 
            # 在开始位置标记tensor的索引  
            start_indices = torch.where(start_mask)[0]  
            # 如果tensor中没有0，直接返回  
            if start_indices.numel() == 0:  
                return None, None  
            # 计算每个0串的长度（通过找到下一个开始位置或tensor的末            
            add0ahead = torch.cat((torch.zeros(1), start_indices))
            add0behind = torch.cat((start_indices, torch.zeros(1)))
            lengths = (add0behind - add0ahead)[:-1]
            # 找到最长0串的长度和对应的索引  
            max_length, max_length_idx = lengths.max(dim=0)  
            # 计算最长0串的起始和结束索引  
            start_index = start_indices[max_length_idx]  
            end_index = start_index + max_length.item() - 1  
            return int(start_index.item()), int(max_length), int(end_index.item())
        
        dataset_len = int(len(dataset))
        val_start_index, val_use_len, _ = find_longest_zeros_segment(dataset.labels)
        
        indices = torch.arange(dataset_len)
        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        return val_subset


    def get_loaders(self, dataset, seed, batch, mode=0, val_ratio=0.1):
        def find_longest_zeros_segment(tensor):  
            # 确保tensor是一维的  
            if tensor.dim() != 1:  
                raise ValueError("Tensor must be 1-dimensional.")  
            # 创建一个标记tensor，其中1表示当前元素是0串的开始，0表示其他情况  
            # 逻辑是：(tensor == 0) & ((tensor[:-1] == 1) | (tensor[1:] == 1)) 的逻辑反  
            # 这里的操作是通过异或（^）来找出0串的开始位置，并且考虑到tensor的首尾  
            start_mask = (tensor[:-1] != tensor[1:]) & (tensor[:-1] == 0) 
            # 在开始位置标记tensor的索引  
            start_indices = torch.where(start_mask)[0]  
            # 如果tensor中没有0，直接返回  
            if start_indices.numel() == 0:  
                return None, None  
            # 计算每个0串的长度（通过找到下一个开始位置或tensor的末            
            add0ahead = torch.cat((torch.zeros(1), start_indices))
            add0behind = torch.cat((start_indices, torch.zeros(1)))
            lengths = (add0behind - add0ahead)[:-1]
            # 找到最长0串的长度和对应的索引  
            max_length, max_length_idx = lengths.max(dim=0)  
            # 计算最长0串的起始和结束索引  
            start_index = start_indices[max_length_idx]  
            end_index = start_index + max_length.item() - 1  
            return int(start_index.item()), int(max_length), int(end_index.item())
        
        dataset_len = int(len(dataset))
        if mode == 0:
            no_val_use_len = int(dataset_len * (1 - val_ratio))
            val_use_len = int(dataset_len * val_ratio)
            val_start_index = random.randrange(no_val_use_len)
        elif mode == 1:
            val_start_index, val_use_len, _ = find_longest_zeros_segment(dataset.labels)
            if val_use_len > int(dataset_len * val_ratio):
                val_use_len = int(dataset_len * val_ratio) 
        
        indices = torch.arange(dataset_len)
        no_val_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        no_val_subset = Subset(dataset, no_val_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)


        train_dataloader = DataLoader(no_val_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, log_path, train_config, test_result, val_result):
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()

        slide_avg_win = train_config['slide_avg_win']
        test_scores, normal_scores = get_full_err_scores(slide_avg_win, test_result, val_result)
        # test_labels = (0.6*np.array(test_labels)).tolist()
        # print(test_labels)
        ratio = train_config['thres_ratio']
        top1_best_info = get_best_performance_data(test_scores, test_labels, ratio, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)



        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print('=========================** Result **============================\n')
        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}')
        print(f'val f1: {info[3]}')
        print(f'auc: {info[4]}\n')

        with open(log_path, 'a') as f:
            print('=========================** Result **============================\n', file=f)
            print(f'F1 score: {info[0]}', file=f)
            print(f'precision: {info[1]}', file=f)
            print(f'recall: {info[2]}', file=f)
            print(f'val f1: {info[3]}', file=f)
            print(f'auc: {info[4]}\n', file=f)
        


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m.%d_%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./logs/{dir_path}/{datestr}.log',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-slide_avg_win', help='window length of slide average', type = int, default=4)
    parser.add_argument('-thres_ratio', help='thres', type = float, default=1)
    parser.add_argument('-fusion_layer_num', help='fusion', type = int, default=2)
    parser.add_argument('-perf_dep', help='Depth of Perfomer', type = int, default=8)
    parser.add_argument('-lr', help='learning rate', type = float, default=0.001)
    parser.add_argument('-loader_mode', help='loader mode', type = int, default=0)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'comment': args.comment,
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'slide_avg_win':args.slide_avg_win,
        'thres_ratio' : args.thres_ratio,
        'fusion_layer_num': args.fusion_layer_num,
        'perf_dep':args.perf_dep,
        'lr':args.lr,
        'loader_mode':args.loader_mode,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)
    main.run()





