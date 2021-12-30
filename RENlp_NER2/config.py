#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: config.py
@time: 2021/12/2 10:21 AM
@desc: 配置类
"""

import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--train_file", default='data/debug/train.txt', type=str)
        parser.add_argument("--eval_file", default='data/debug/dev.txt', type=str)
        parser.add_argument("--test_file", default='data/debug/test.txt', type=str)
        parser.add_argument("--model_name_or_path", default='bert-base-chinese', type=str)
        parser.add_argument("--output_dir", default='data/output', type=str)
        parser.add_argument("--my_model_dir", default='data/my_model', type=str)

        ## other parameters
        parser.add_argument("--config_name", default="bert-base-chinese/config.json", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="bert-base-chinese/vocab.txt", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--model_name", default="bert-base-chinese/pytorch_model.bin", type=str,
                            help="Prtetrained model name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="data/cache", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")

        parser.add_argument("--max_seq_length", default=128, type=int)
        parser.add_argument("--do_train", default=True, type=boolean_string)
        parser.add_argument("--do_eval", default=True, type=boolean_string)
        parser.add_argument("--do_test", default=True, type=boolean_string)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=8, type=int)
        parser.add_argument("--learning_rate", default=3e-5, type=float)
        parser.add_argument("--num_train_epochs", default=40, type=float)
        parser.add_argument("--warmup_proprotion", default=0.1, type=float)
        parser.add_argument("--use_weight", default=1, type=int)
        parser.add_argument("--local_rank", type=int, default=-1)
        parser.add_argument("--seed", type=int, default=2019)
        parser.add_argument("--fp16", default=False)
        parser.add_argument("--loss_scale", type=float, default=0)
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--do_lower_case", default=True, action='store_true')
        parser.add_argument("--logging_steps", default=500, type=int)
        parser.add_argument("--clean", default=True, type=boolean_string, help="clean the output dir")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("--need_birnn", default=True, type=boolean_string)
        parser.add_argument("--rnn_dim", default=256, type=int)

        parser.add_argument("--ex_index", default=0, type=int,
                            help="打印预处理后的数据条目数，默认为0不打印")
        parser.add_argument('--log_path', type=str, default='data/bert_ner_log',
                            help='Path of bert_ner_log')

        self.args = parser.parse_args()


config = Config()
