import torch
import torch.nn as nn

torch.manual_seed(1234)

from RENlp_NER1.util import argmax, log_sum_exp
import RENlp_NER1.util as util


class BiLSTM_CRF(nn.Module):
    """
    自定义一个模型——通过继承nn.Module类来实现,并实现forward方法
    """

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout):
        super(BiLSTM_CRF, self).__init__()  # 调用父类的init
        self.embedding_dim = embedding_dim  # 嵌入维度：嵌入向量的维度，即用多少维来表示一个符号（这里定义的维度是300）
        self.hidden_dim = hidden_dim  # LSTM中隐藏层的维度
        self.vocab_size = vocab_size  # 词汇量大小
        self.tag_to_ix = tag_to_ix  # 标签转下标的词典
        self.tagset_size = len(tag_to_ix)  # 输出维度：目标取值范围大小
        self.dropout = dropout

        ''' 
        Embedding的用法
        A simple lookup table that stores embeddings of a fixed dictionary and size.
        This module is often used to store word embeddings and retrieve them using indices. 
        The input to the module is a list of indices, and the output is the corresponding word embeddings.
        requires_grad: 用于说明当前量是否需要在计算中保留对应的梯度信息
        '''
        # an Embedding module containing 词汇量大小的 tensors of size 词向量维度
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  # 固定大小的词典的嵌入向量的查找表

        # LSTM模型
        '''
        默认参数意义：input_size，hidden_size，num_layers
        hidden_size : LSTM在运行时里面的维度。隐藏层状态的维数，即隐藏层节点的个数
        torch里的LSTM单元接受的输入都必须是3维的张量(Tensors):
           第一维体现的每个句子的长度，即提供给LSTM神经元的每个句子的长度，如果是其他的带有带有序列形式的数据，则表示一个明确分割单位长度，
           第二维度体现的是batch_size，即每一次给网络句子条数
           第三维体现的是输入的元素，即每个具体的单词用多少维向量来表示
        '''
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=2, bidirectional=True,
                            dropout=self.dropout)  # num_layers：循环神经网络的层数，bidirectional默认是false，代表不用双向LSTM

        # Maps the output of the LSTM into tag space.
        # 建立一个把LSTM的输出到标签空间的映射关系，通过一个线性连接层将 BiLSTM 隐状态维度 转变为 tag 的种类大小
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 转移矩阵是随机的，在网络中会随着训练不断更新
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer to the start tag and we never transfer
        # from the stop tag
        # 转移矩阵： 列标 转 行标
        # 规定：其他tag不能转向start，stop也不能转向其他tag
        self.transitions.data[tag_to_ix[util.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[util.STOP_TAG]] = -10000

        # 初始化hidden layer
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(4, 1, self.hidden_dim // 2),
                torch.randn(4, 1, self.hidden_dim // 2))

    # 前向算法：feats是LSTM所有时间步的输出
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # alpha初始为-10000
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        # start位置的alpha为0
        init_alphas[0][self.tag_to_ix[util.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 包装进变量，实现自动反向传播
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            # 当前 timestep 的前向tensor
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                '''
                LSTM生成的矩阵是emit score[观测/发射概率], 即公式中的H()函数的输出
                CRF是判别式模型
                emit score: BilSTM 对序列中 每个位置 的 对应标签 打分的和
                transition score 是该序列状态转移矩阵中对应的和
                Score = EmissionScore + TransitionScore
                '''
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[util.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    # LSTM的输出, 即emit score
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        embeds = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[util.START_TAG]], dtype=torch.long), tags])

        # feats 是bilstm提取的特征[标签数量长度]
        # 转移+前向
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[util.STOP_TAG], tags[-1]]
        return score

    # Decoding的意义：
    # 给定一个已知的观测序列，求其最有可能对应的状态序列
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[util.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[util.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        # 回退找路
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        # 去掉start
        start = best_path.pop()
        assert start == self.tag_to_ix[util.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # tags 是句子对应的tag,长度与句子等长
    def neg_log_likelihood(self, sentence, tags):
        # feats 为LSTM提取的标签对应值
        feats = self._get_lstm_features(sentence)
        # 前向算法分数
        forward_score = self._forward_alg(feats)
        # 真实分数
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    # 重写 原module里的 forward
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return lstm_feats, score, tag_seq
