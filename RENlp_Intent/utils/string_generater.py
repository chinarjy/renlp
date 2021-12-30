# -*- coding = utf-8 -*-
# @Time :2021/10/21 9:54
# @Author:ren.jieye
# @Describe:根据正则表达式生成字符串
# @File : string_generater.py
# @Software: PyCharm IT
from xeger import Xeger
from segByJieba import sent2word

def str_generate_by_regex(regex):
    _x = Xeger()
    testStrs = []
    for i in range(30):
        testStr = _x.xeger(regex)
        testStrs.append(testStr)
    return testStrs


if __name__ == '__main__':
    with open('../data/ID4正则.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    tags = []
    sen_tag = {}
    for line in lines:
        line = line.replace('/n', '')
        regex = line.split('***')[0]
        tag = line.split('***')[1]
        strs = str_generate_by_regex(regex)
        for str in strs:
            sen_tag[str] = tag
    with open('../data/train.txt', 'a', encoding='utf-8') as train_file:
        for key in sen_tag.keys():
            print('****************************')
            # print(key)
            # show = ''
            # for word in sent2word(key):
            #     show += word + ' '
            print('{' + key + '} : ' + sen_tag[key])
            train_file.write(key + '***' + sen_tag[key])
    f.close()
    train_file.close()

    # str_generate_by_regex('[Ii][Qq].{0,2}(Light)|([Ll][Ee]?[Dd]|[Ll][Ee]?[Dd])((大|打|尾)灯)?.{0,5}(造型美观|亮度高|照射距离远|能耗低|寿命长|矩阵(的|式)?)|((大|打|尾)灯.{0,4}是|矩阵式).{0,2}([Ll][Ee]?[Dd]|[Ll][Ee]?[Dd])(大灯)?|(贯穿|矩阵式?).{0,4}(大|尾)灯|(大|尾)灯.{0,4}(矩阵式)|拥有了?灵动的?双眼')