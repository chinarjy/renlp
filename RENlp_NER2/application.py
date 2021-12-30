#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: application.py
@time: 2021/12/2 10:21 AM
@desc: 可以封装成api接口，实现单个句子的实体识别
"""
import os
import pickle
from RENlp_NER2.models import BERT_BiLSTM_CRF
from RENlp_NER2.logger import logger
from RENlp_NER2.config import config
import torch
from transformers import (BertConfig, BertTokenizer)
from RENlp_NER2.utils import NerProcessor, get_Dataset
import torch.nn.functional as F
from tqdm import tqdm, trange
from nltk import word_tokenize
from RENlp_NER2.utils import NerProcessor

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)


class entitys(object):
    entity_name = ''
    entity_type = ''
    start_position = -1
    end_position = -1


def tokenize(tokenizer, text: str):
    """ tokenize input"""
    # words = []
    # for w in text:
    #     w1 = word_tokenize(w)
    #     words.append(w1)
    words = word_tokenize(text)[0]
    tokens = []
    valid_positions = []
    for i, word in enumerate(words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i in range(len(token)):
            if i == 0:
                valid_positions.append(1)
            else:
                valid_positions.append(0)
    return tokens, valid_positions


def preprocess(tokenizer, args, text: str):
    """ preprocess """
    tokens, valid_positions = tokenize(tokenizer, text)
    ## insert "[CLS]"
    tokens.insert(0, "[CLS]")
    valid_positions.insert(0, 1)
    ## insert "[SEP]"
    tokens.append("[SEP]")
    valid_positions.append(1)
    segment_ids = []
    for i in range(len(tokens)):
        segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < args.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        valid_positions.append(0)
    return input_ids, input_mask, segment_ids, valid_positions


def predict(text):
    args = config.args

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    tokenizer = BertTokenizer.from_pretrained(args.my_model_dir, do_lower_case=args.do_lower_case)
    args = torch.load(os.path.join(args.my_model_dir, 'training_args.txt'))
    model = BERT_BiLSTM_CRF.from_pretrained(args.my_model_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
    model.to(device)

    processor = NerProcessor()
    test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="text", text=text)

    logger.info("***** Running test *****")
    logger.info(f" Num examples = {len(test_examples)}")
    logger.info(f" Batch size = {args.eval_batch_size}")

    all_ori_tokens = [f.ori_tokens for f in test_features]
    all_ori_labels = [e.label.split(" ") for e in test_examples]
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    model.eval()

    pred_labels = []

    label_list = processor.get_labels(args)
    if os.path.exists(os.path.join(args.output_dir, "label2id.pkl")):
        with open(os.path.join(args.output_dir, "label2id.pkl"), "rb") as f:
            label2id = pickle.load(f)
    else:
        label2id = {l: i for i, l in enumerate(label_list)}
        with open(os.path.join(args.output_dir, "label2id.pkl"), "wb") as f:
            pickle.dump(label2id, f)
    id2label = {value: key for key, value in label2id.items()}
    for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model.predict(input_ids, segment_ids, input_mask)
        # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        # logits = logits.detach().cpu().numpy()

        for l in logits:

            pred_label = []
            for idx in l:
                pred_label.append(id2label[idx])
            pred_labels.append(pred_label)

    assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)
    print(len(pred_labels))

    for index, (ori_tokens, prel) in enumerate(zip(all_ori_tokens, pred_labels)):
        entitys = []
        print('*********************')
        entity = ''
        type = ''
        start_pos = 0
        end_pos = len(prel)
        for i, (ot, pl) in enumerate(zip(ori_tokens, prel)):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            else:
                # print(f"{ot} {pl}")
                if i > 0 and prel[i - 1] == 'O' and pl != 'O' and pl.split('-')[0] != 'B':
                    pl = 'B' + '-' + pl.split('-')[1]
                elif i > 0 and prel[i -1] != 'O' and pl != 'O' and pl.split('-')[1] != prel[i-1].split('-')[1]:
                    pl = 'B' + '-' + pl.split('-')[1]
                print(f"{ot} {pl}")
            if pl != 'O' and pl.split('-')[0] == 'B':
                entity = ot
                type = pl.split('-')[1]
                start_pos = i
            elif pl != 'O' and pl.split('-')[0] == 'I' and pl.split('-')[1] == type and i < len(prel) - 1 and prel[i + 1] != 'O' and pl.split('-')[1] == prel[i+1].split('-')[1]:
                entity = entity + ot
                type = type
            elif pl != 'O' and (i == len(prel) or prel[i + 1] == 'O'):
                entity = entity + ot
                type = type
                end_pos = i
                entitys.append({
                    'text': text[index],
                    'entity': entity,
                    'type': type,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })
                entity = ''
                type = ''
            elif pl != 'O' and pl.split('-')[1] != prel[i+1].split('-')[1]:
                entity = entity + ot
                type = type
                end_pos = i
                entitys.append({
                    'text': text[index],
                    'entity': entity,
                    'type': type,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })
                entity = ''
                type = ''
        print(entitys)
    # with open(os.path.join(args.output_dir, "token_labels_.txt"), "w", encoding="utf-8") as f:
    #     for ori_tokens, ori_labels, prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
    #         for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
    #             if ot in ["[CLS]", "[SEP]"]:
    #                 continue
    #             else:
    #                 f.write(f"{ot} {ol} {pl}\n")
    #         f.write("\n")


if __name__ == '__main__':
    text = ['北京到上海的飞机明天将全部停飞',
            '任杰也同学是大连理工大学的优秀毕业生，现在在长春工作',
            '国务院总理李鹏于今天上午10时在北京出席了人大常委会第三次会议。',
            '中国共产党第十九届中央委员会第六次全体会议于11月8日至11日在北京召开，重点研究全面总结了党的百年奋斗的重大成就和历史经验问题。',
            '在山东济宁市金乡县王杰纪念馆内，情景再现剧《英雄壮举》通过幻影成像等技术，把人们的思绪拉回到1965年7月14日。',
            '王柏蓉出生在长岭县，在会和工业有限公司工作',
            '王百菊在阜新市工作，工作单位是畜牧局',
            '国家主席习近平在北京向2021年国际会议（广州）开幕式发表视频致辞。',
            '中共中央总书记、国家主席习近平将于12月3日同老挝人民革命党中央总书记、国家主席通伦举行视频会晤，并以视频连线形式共同见证中老铁路通车'
            ]
    predict(text)
