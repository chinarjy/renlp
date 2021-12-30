import RENlp_NER1.util as util
import re
import torch
from RENlp_NER1.models.bistmCRF import BiLSTM_CRF

'''
模型预测
'''
EMBEDDING_DIM = util.EMBEDDING_DIM
HIDDEN_DIM = util.HIDDEN_DIM
TRAIN_DATA_PATH = util.TRAIN_DATA_PATH
MODEL_PATH = util.MODEL_PATH
NEWS_PATH = util.NEWS_PATH
MODEL_PATH = util.MODEL_PATH
MODEL_PRO_PATH = util.MODEL_PRO_PATH
model = {
    'BiLSTM': MODEL_PATH,
    'BiLSTMPro': MODEL_PRO_PATH
}


def cleanSent(sent):
    sent = re.sub(r"\n", ",", sent)
    sent = re.sub(r"\t", "", sent)
    return sent


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def cut_sentence(data):
    sentLst = data.split("。")
    return sentLst


# 模型选择
MODEL = model['BiLSTM']

_, word_to_ix, tag_to_ix = util.data_prepare(TRAIN_DATA_PATH)
OrderedDict = torch.load(MODEL)
len_ = len(OrderedDict["word_embeds.weight"])

model = BiLSTM_CRF(len_, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, dropout=0.5)
model.load_state_dict(torch.load(MODEL))
model.eval()

ix_to_tag = {v: k for k, v in tag_to_ix.items()}


# for parent, dirnames, filenames in os.walk(NEWS_PATH):   #  对目录进行遍历
#     for filename in filenames:
#         file_path_in = os.path.join(parent, filename)
#         print("transforming.....", file_path_in)
#         file_path_out = r"NERout\ner_{}".format(filename)
#         with open(file_path_in, 'r', encoding='utf-8') as f:
#             test_d = f.read()
#         # test_d = codecs.open(file_path_in, 'r', encoding='utf-8').read()
#         test_d = strQ2B(test_d)
#         test_d = re.sub(r"\n\n", "", test_d)
#         test_d_Lst = cut_sentence(test_d)
#         result = set()
#
#         for sent in test_d_Lst:
#             sent = cleanSent(sent)
#             if len(sent) < 4: continue
#             test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in sent]
#             test_res_ = torch.tensor(test_id, dtype=torch.long)
#             eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]
#
#             tag2word_res = util.tag2word(sent, eval_res)
#             for s in tag2word_res:
#                 result.add(s)
#
#         with codecs.open(file_path_out, 'w+', 'utf-8') as surveyp:
#             surveyp.write(",\n".join(result))

def predict(sent):
    test_sent = sent
    test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in test_sent]
    test_res_ = torch.tensor(test_id, dtype=torch.long)
    eval_res = [ix_to_tag[t] for t in model(test_res_)[2]]
    print(eval_res)
    entitys = []
    types = []
    entity = ''
    for i in range(len(test_sent)):
        if eval_res[i] != 'O':
            tag = eval_res[i].split('_')[0]
            type = eval_res[i].split('_')[1]
            if tag == 'B':
                entity = test_sent[i]
                start_pos = i
            elif tag == 'M':
                entity += test_sent[i]
            elif tag == 'E':
                entity += test_sent[i]
                # entitys.append(entity)
                end_pos = i
                # types.append(type)
                entitys.append({
                    'entity': entity,
                    'type': type,
                    'start_position': start_pos,
                    'end_position': end_pos
                })
                entity = ''
    for i, entity in enumerate(entitys):
        test_sent = test_sent + '#' + entity['entity'] + '#' + entity['type']
    print(test_sent)
    print(entitys)
    return test_sent


if __name__ == '__main__':
    predict('玄幻小说《佣兵天下》的作者是说不得大师，自2002年起连载于书旗小说网')

# # #['O', 'O', 'O', 'O', 'B_PERSON', 'E_PERSON', 'O', 'B_PERSON', 'E_PERSON', 'O', 'O', 'B_LOCATION', 'E_LOCATION', 'O', 'B_LOCATION', 'M_LOCATION', 'M_LOCATION', 'E_LOCATION', 'O', 'O', 'O', 'O', 'O', 'O']
