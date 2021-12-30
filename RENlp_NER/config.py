# -*- coding = utf-8 -*-
# @Time :2021/11/25 8:59
# @Author:ren.jieye
# @Describe:
# @File : config.py
# @Software: PyCharm IT
import argparse


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--data_dir",
                            default="data/debug",
                            type=str,
                            # required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("--bert_model", default="bert-base-chinese", type=str,
                            # required=True,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                                 "bert-base-multilingual-cased, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default="ner",
                            type=str,
                            # required=True,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default="data/output",
                            type=str,
                            # required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")

        ## Other parameters
        parser.add_argument("--cache_dir",
                            default="",
                            type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        parser.add_argument("--do_train",
                            default=True,
                            action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval",
                            default=True,
                            action='store_true',
                            help="Whether to run eval or not.")
        parser.add_argument("--eval_on",
                            default="dev",
                            help="Whether to run eval on the dev set or test set.")
        parser.add_argument("--do_lower_case",
                            action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=1e-3,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=1.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10% of training.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--fp16',
                            action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument('--loss_scale',
                            type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

        parser.add_argument('--log_path', type=str, default='data/bert_ner_log.txt', help='Path of bert_ner_log')

        parser.add_argument('--train_file', type=str, default='data/debug/train.txt',
                            help='Train data for bert_ner')
        parser.add_argument('--dev_file', type=str, default='data/debug/dev.txt',
                            help='Dev data for bert_ner')
        parser.add_argument('--test_file', type=str, default='data/debug/test.txt',
                            help='Test data for bert_ner')
        parser.add_argument('--sep', type=str, default=' ',
                            help='数据集中句子的分割符')
        self.args = parser.parse_args()



config = Config()