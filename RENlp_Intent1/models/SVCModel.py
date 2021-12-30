# -*- coding = utf-8 -*-
# @Time :2021/10/25 10:37
# @Author:ren.jieye
# @Describe:
# @File : SVCModel.py
# @Software: PyCharm IT

import re
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from scipy import sparse, io
import jieba


class SVCModel:
    """
    该类将所有模型训练、预测、数据预处理、意图识别的函数包括其中
    """

    # 初始化模块
    def __init__(self):
        self.stopwords = {}
        # 加载停用词词典
        with open(r'../data/中文停用词词典.txt', 'r', encoding='utf-8') as fr:
            for word in fr:
                self.stopwords[word.strip()] = 0
        self.model = ""  # 成员变量，用于存储模型
        self.vectorizer = ""  # 成员变量，用于存储tfidf统计值

    # 训练模块
    def train(self):
        """
        训练结果存储在成员变量中，没有return
        """
        # 从txt文件读取训练样本
        # d_train = pd.read_excel("../data/20210813_idcc.xlsx", sheet_name='Sheet1')
        sentences = []
        tags = []
        with open('../data/train.txt', 'r', encoding='utf-8') as train_file:
            lines = train_file.readlines()
        # 对训练样本进行预处理
        for line in lines:
            line = line.replace('\n', '')
            print(line)
            sen = self.fun_clean(line.split('***')[0])
            sentences.append(sen)
            tags.append(line.split('***')[1])
        # for i in range(len(sentences)):
        #     print(sentences[i] + ' : ' + tags[i])
        print("训练样本 = %d" % len(sentences))
        # 利用sklearn中的函数进行tfidf训练
        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        # # 注意，这里自己指定token_pattern，否则sklearn会自动将一个字长度的单词过滤筛除
        features = self.vectorizer.fit_transform(sentences)
        print("训练样本特征表长度为 " + str(features.shape))
        # print(self.vectorizer.get_feature_names())
        # 使用逻辑回归进行训练和预测
        self.model = SVC(C=20)
        self.model.fit(features, tags)
        intent_model_path = '../data/intent_model_SVC.pkl'
        with open(intent_model_path, 'wb') as model:
            pickle.dump(self.model, model)
        intent_vectorizer_path = '../data/intent_vectorizer_SVC.pkl'
        with open(intent_vectorizer_path, 'wb') as vectorizer:
            pickle.dump(self.vectorizer, vectorizer)

    # 预处理函数
    def fun_clean(self, sentence):
        """
        预处理函数
        :输入 用户输入语句:
        :输出 预处理结果:
        """
        jieba.load_userdict('../data/一汽大众领域词典.txt')
        word_lst = [w for w in list(jieba.cut(sentence)) if w not in self.stopwords]
        output_str = ' '.join(word_lst)
        output_str = re.sub(r'\s+', ' ', output_str)
        return output_str.strip()

    # 分类主函数
    def fun_clf(self, sentence):
        """
        意图识别函数
        :输入 用户输入语句:
        :输出 意图类别，分数:
        """
        # 对用户输入进行预处理
        sentence = self.fun_clean(sentence)
        # 得到意图分类结果（0为“查询”类别，1为“订票”类别，2为“终止服务”类别）
        clf_result, score = self.predict_model(sentence)  # 使用训练的模型进行意图预测
        # clf_result, score = self.predict_rule(sentence)  # 使用规则进行意图预测（可与用模型进行意图识别的方法二选一）
        return clf_result, score