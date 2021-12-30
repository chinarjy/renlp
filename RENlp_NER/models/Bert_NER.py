# -*- coding = utf-8 -*-
# @Time :2021/11/25 10:36
# @Author:ren.jieye
# @Describe:
# @File : Bert_NER.py
# @Software: PyCharm IT
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize
import json
import os
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  get_linear_schedule_with_warmup)


class Bert_NER(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class MyNER:

    def __init__(self, model_dir: str):
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        # model_dir = model_dir+'/pytorch_model.bin'
        # print(model_dir)
        model = Bert_NER.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir+'/vocab.txt', do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        # words = []
        # for w in text:
        #     w1 = word_tokenize(w)
        #     words.append(w1)
        words = word_tokenize(text)[0]
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids, input_mask, segment_ids, valid_positions

    def predict(self, text: str):
        input_ids, input_mask, segment_ids, valid_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        valid_ids = torch.tensor([valid_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask, valid_ids=valid_ids)
        logits = F.softmax(logits, dim=2)
        logits_label = torch.argmax(logits, dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values, label in zip(logits[0], logits_label)]

        logits = []
        pos = 0
        for index, mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index - pos], logits_confidence[index - pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label], confidence) for label, confidence in logits]
        words = word_tokenize(text)[0]
        assert len(labels) == len(words)
        output = [{"word": word, "tag": label, "confidence": confidence} for word, (label, confidence) in
                  zip(words, labels)]
        return output
