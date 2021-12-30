import torch
import pandas as pd
import os
import re

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK_TAG = "<UNK>"

EPOCHES = 100

EMBEDDING_DIM = 300
HIDDEN_DIM = 200
BATCH_SIZE = 128
lr = 0.0001

TRAIN_DATA_PATH = os.path.join(os.path.abspath('.') + r'/data/train.csv')
# TRAIN_DATA_PATH = os.path.abspath('.') + '\\PyTorch_NER\\data\\train.csv'
MODEL_PRO_PATH = os.path.join(r'model_files',
                          "bilstm_model_emb_{}_hidden_{}_batch_{}.pth".format(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE))
MODEL_PATH = os.path.join(os.path.abspath('.') + r'/model_files',
                          "bilstm_crf_model_emb_{}_hidden_{}_batch_{}.pth".format(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE))
NEWS_PATH = r"../news"


def data_prepare(TRAIN_DATA_PATH):
    # training data
    print(os.path.abspath('utils'))
    print(TRAIN_DATA_PATH)

    df = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    df.char = df['char'].apply(lambda x: eval(x))  # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    df.tag = df['tag'].apply(lambda y: eval(y))

    word_to_ix = {}  # 字典类型：存储数据集中的字及对应的实体标签（key：字，value：实体标签）
    for i in range(len(df)):
        sentence, _ = df.iloc[i]  # iloc函数：通过行号来取行数据（如取第二行的数据）
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {START_TAG: 0, STOP_TAG: 1, UNK_TAG: 2, 'O': 3}
    tag_to_ix_tmp = dict(
        zip([a + b for a in ['B', 'M', 'E'] for b in ['_人物', '_城市', '_国家', '_网站', '_歌曲', '_影视作品', '_学校', '_电视综艺', '_企业', '_地点', '_音乐专辑', '_网络小说', '_景点']], range(4, 43)))
    tag_to_ix.update(tag_to_ix_tmp)  # 字典类型，存储实体标签对应的索引（例如B_PERSON：5）

    return df, word_to_ix, tag_to_ix


#  返回vec中每一行最大的那个元素的下标
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


#  单词转为索引
def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w not in to_ix.keys():
            w = UNK_TAG
        idxs.append(to_ix[w])
    return torch.tensor(idxs, dtype=torch.long)
    # return idxs


# 计算一维向量vec 与 其最大值的   log_sum_exp
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# for eval
def tag2word(sent, tagLst):
    res = []
    for i, t in enumerate(tagLst):
        flag = t.split('_')
        if len(flag) > 1:
            if flag[0] == "E":
                tagLst[i] = sent[i] + tagLst[i] + "|"
            else:
                tagLst[i] = sent[i] + tagLst[i]
        else:
            tagLst[i] = '|'
    tmp = "".join(tagLst)
    tmp = set(tmp.split("|"))

    for w in tmp:
        if w:
            w_w = re.findall("[0-9]?[\u4e00-\u9fa5]?", w)
            w_w = "".join(w_w)
            w_t = w.split("_")[-1]
        else:
            continue
        res.append(w_w + " " + w_t)

    return res


if __name__ == '__main__':
    print(TRAIN_DATA_PATH)
