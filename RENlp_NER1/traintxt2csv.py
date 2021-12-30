# -*- coding = utf-8 -*-
# @Time :2021/11/1 8:44
# @Author:ren.jieye
# @Describe:
# @File : traintxt2csv.py
# @Software: PyCharm IT
import codecs
import csv
import pandas as pd


def load_csv(file):
    data_list = []
    with codecs.open(file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            data = list(line.values())
            data_list.append(data)
    return data_list


if __name__ == '__main__':
    data_list = load_csv('data/traintxt.csv')
    char_list = []
    tag_list = []
    head_types = []
    tail_types = []
    for data in data_list:
        sentence = data[0]
        head = data[2]
        head_type = data[3]
        head_index = int(data[4])
        tail = data[5]
        tail_type = data[6]
        tail_index = int(data[7])
        head_types.append(head_type)
        tail_types.append(tail_type)
        chars = []
        tags = []
        # print('{}, {}, {}, {}, {}, {}, {}'.format(sentence, head, head_type, head_index, tail, tail_type, tail_index))
        show = ''
        for i in range(len(sentence)):
            char = sentence[i]
            if i == head_index:
                tag = 'B_' + head_type
            elif head_index < i < len(head) + head_index - 1:
                tag = 'M_' + head_type
            elif i == len(head) + head_index - 1:
                tag = 'E_' + head_type
            elif i == tail_index:
                tag = 'B_' + tail_type
            elif tail_index < i < len(tail) + tail_index - 1:
                tag = 'M_' + tail_type
            elif i == len(tail) + tail_index - 1:
                tag = 'E_' + tail_type
            else:
                tag = 'O'
            chars.append(char)
            tags.append(tag)
        char_list.append(chars)
        tag_list.append(tags)
    train_df = pd.DataFrame()

    train_df['char'] = char_list
    train_df['tag'] = tag_list

    train_df.to_csv(r"../data/train.csv", encoding='utf-8')

    print(set(head_types + tail_types))
