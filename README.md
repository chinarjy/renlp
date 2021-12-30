# RENlp

#### 介绍
REN NLP实现

#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明
Python + PyTorch实现nlp常用任务
RENlp_Sentiment: 情感分析
1. 可以对VOC和外卖评价进行分析，voc分成正向、中性和负向三个情感，外卖评价分为positive和negative两个情感
2. 对不同的场景进行训练和预测需要调整配置，config.py:type 参数要调整，不同的场景数据保存在不同的文件夹，模型通过名称进行区分
3. 这两个场景都设置成了3分类问题，外卖评价实际是2分类，out_dim设置成3不影响模型训练和结果预测
4. 测试结果：car_Sentiment_CNN_epoch2_1206_11_33_56.pth
               precision      recall    f1-score    support
         None       0.00      0.00      0.00         0
          中性       0.72      0.92      0.81      4969
          正向       0.44      0.15      0.22      1071
          负向       0.00      0.00      0.00      1161
     accuracy                           0.66      7201
    macro avg       0.29      0.27      0.26      7201
 weighted avg       0.57      0.66      0.59      7201

RENlp_Intent: 意图识别——基于CNN实现试乘试驾语音质检功能
RENlp_NRE：关系抽取——基于CNN实现实体关系抽取
RENlp_NER: 实体识别——基于bert+crf实现命名实体识别
RENlp_NER1: 实体识别——基于BiLSTM+CRF实现命名实体识别
RENlp_NER2： 实体识别——基于Bert+BiLSTM+CRF实现命名实体识别
RENlp_PosTag： 词性标注，tag_demo.py: 基于逻辑回归进行监督学习实现英文词性标注， tag_jieba.py: 基于jieba实现词性标注
RENlp_SegWord： 分词， 自编代码实现基于词典的分词（正向、负向、双向最大匹配法），使用jieba进行分词


#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


