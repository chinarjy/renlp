#!/usr/bin/env python
# encoding: utf-8
"""
@author: jieye.ren
@contact: chinarjy@163.com
@software: pycharm
@file: predict.py
@time: 2021/12/3 11:04 AM
@desc: Intent 预测【机器学习：logistics】- model：intent_model.pkl ， vocab： intent_vectorizer.pkl
"""

import re
import pickle
import jieba
import argparse
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request


# app = Flask(__name__)


# 定义类
class logistic_model:
    """
    该类将所有模型训练、预测、数据预处理、意图识别的函数包括其中
    """

    # 初始化模块
    def __init__(self):
        intent_model_path = 'data/output/intent_model.pkl'
        self.model = pickle.load(open(intent_model_path, "rb"))  # 成员变量，用于存储模型
        intent_vectorizer_path = 'data/output/intent_vectorizer.pkl'
        self.vectorizer = pickle.load(open(intent_vectorizer_path, "rb"))  # 成员变量，用于存储tfidf统计值

    # 预测模块（使用模型预测）
    def predict_model(self, sentence):
        sent_features = self.vectorizer.transform([sentence])  # tfidf
        pre_test_label = self.model.predict(sent_features)  # 逻辑回归
        # pre_test = self.model.predict_proba(sent_features).tolist()[0]
        # clf_result = pre_test.index(max(pre_test))
        # score = max(pre_test)
        scores = self.model.decision_function(sent_features)
        index = scores.argmax(axis=1)
        score = scores[0][index]
        return pre_test_label, score

    # 预处理函数
    def fun_clean(self, sentence):
        # -----------------------------------------------------
        # 加载停用词词典
        stopwords = {}
        with open(r'data/中文停用词词典.txt', 'r', encoding='utf-8') as fr:
            for word in fr:
                stopwords[word.strip()] = 0

        # -----------------------------------------------------
        # 分词，并去除停用词
        jieba.load_userdict('data/一汽大众领域词典.txt')
        word_lst = [w for w in list(jieba.cut(sentence)) if w not in stopwords]
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
        # sentence = self.fun_clean(sentence)
        # 得到意图分类结果
        clf_result, score = self.predict_model(sentence)  # 使用训练的模型进行意图预测
        return clf_result, score


# @app.route('/getIntent', methods=['post'])
def main():
    parser = argparse.ArgumentParser(description='意图识别')  # 创建解析器
    parser.add_argument('--sen_pre', type=bool, default=False, help='测试方式：True-单句测试， False-文档测试')  # 添加参数
    args = parser.parse_args()  # 解析参数
    predict_way = args.sen_pre
    model = logistic_model()
    if predict_way:
        # 对用户输入进行意图识别
        sentence = '静音型还是比较好的，不是双玻璃，但是后排还是隐私玻璃，不是双层的，但是它的隔音效果其实还是比较好的。'
        # sentence = request.json.get('question')  # 通过文本以json格式传入参数
        label, score = model.fun_clf(model.fun_clean(sentence))
        print(sentence+'- label : {} , score : {}'.format(label, score))
    else:
        with open('data/origin/test.txt', 'r', encoding='utf-8') as test_file:
            lines = test_file.readlines()
        sentences = []
        t_tags = []
        p_tags = []
        with open('data/result/result_logistic.txt', 'w', encoding='utf-8') as result_file:
            for line in lines:
                line = line.strip()
                sentence = line.split('***')[0]
                t_tag = line.split('***')[1]
                sentences.append(sentence)
                t_tags.append(t_tag)
                label, score = model.fun_clf(model.fun_clean(sentence))
                # p_tag = label[0]
                r = score[0] > 2.5
                if r:
                    p_tag = label[0]
                else:
                    p_tag = 'None'
                p_tags.append(p_tag)
                result_file.write(line+'***'+p_tag+'\n')
                print(line + '***' + p_tag)

        test_file.close()
        result_file.close()
        print('**********************************************')
        from sklearn.metrics import classification_report
        print(classification_report(t_tags, p_tags))
        # print('*****************None不预测********************')
        # t_tags_n = []
        # p_tags_n = []
        # for i in range(len(t_tags)):
        #     if t_tags[i] != 'None':
        #         t_tags_n.append(t_tags[i])
        #         p_tags_n.append(p_tags[i])
        # print(classification_report(t_tags_n, p_tags_n))


if __name__ == '__main__':
    main()


