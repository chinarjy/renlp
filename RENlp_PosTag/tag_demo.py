# -*- coding = utf-8 -*-
# @Time :2021/8/27 14:33
# @Author:ren.jieye
# @Describe: 词性标注
# @File : tag_demo.py
# @Software: PyCharm IT
import numpy as np

tag2id, id2tag = {}, {}  # maps tag to id  , tag2id:{"VB":0,"NNP":1,......}, id2tag:{0:"VB",1:"nnp",......}
word2id, id2word = {}, {}  # maps word to id

for line in open('data/traindata.txt'):
    items = line.split('/')
    word, tag = items[0], items[1].rstrip()  # 抽取每一行里的单词和词性
    # print(word, '---', tag)
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag
M = len(word2id)  # M：词典的大小  number of words in dictionary
N = len(tag2id)   # N：词性的种类个数 number of tags in tag set

# 构建pi、A、B
pi = np.zeros(N)  # 每个词性出现在句子中第一个位置的概率,  N: # of tags  pi[i]: tag i出现在句子中第一个位置的概率
A = np.zeros([N, M])  # A[i][j]: 给定tag i, 出现单词j的概率。 N: # of tags M: # of words in dictionary
B = np.zeros([N, N])  # B[i][j]: 之前的状态是i, 之后转换成转态j的概率 N: # of tags\n
prev_tag = ""
for line in open('data/traindata.txt'):
    items = line.split('/')
    wordId, tagId = word2id[items[0]], tag2id[items[1].rstrip()]
    if prev_tag == "":
        pi[tagId] += 1
        A[tagId][wordId] += 1
    else:
        A[tagId][wordId] += 1
        B[tag2id[prev_tag]][tagId] += 1
    if items[0] == '.':
        prev_tag = ''
    else:
        prev_tag = items[1].rstrip()
pi = pi/sum(pi)
# print(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])
# print(A[1])
# print(B[1])
#  到此为止计算完了模型的所有的参数： pi, A, B

def log(v):
    if v == 0:
        return np.log(v+0.000001)
    return np.log(v)

def viterbi(x, pi, A, B):
    x = [word2id[word] for word in x.split(' ')]
    T = len(x)

    dp = np.zeros([T, N])
    ptr = np.array([[0 for x in range(N)] for y in range(T)])

    for j in range(N):
        dp[0][j] = log(pi[j]) + log(A[j][x[0]])

    for i in range(1, T):
        for j in range(N):
            dp[i][j] = -99999
            for k in range(N):
                score = dp[i - 1][k] + log(B[k][j]) + log(A[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k
    # decoding: 把最好的tag sequence 打印出来
    best_seq = [0]*T
    # step1: 找出对应于最后一个单词的词性
    best_seq[T-1] = np.argmax(dp[T-1])
    # step2: 通过从后到前的循环来依次求出每个单词的词性
    for i in range(T-2, -1, -1):
        best_seq[i] = ptr[i+1][best_seq[i+1]]
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])

x = 'I am going to read this book in the flight'
viterbi(x, pi, A, B)


