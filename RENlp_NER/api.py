# -*- coding = utf-8 -*-
# @Time :2021/11/25 14:09
# @Author:ren.jieye
# @Describe:
# @File : api.py
# @Software: PyCharm IT
from flask import Flask, request, jsonify
from flask_cors import CORS
from RENlp_NER.data_prcessor import InputExample, InputFeatures, NerProcessor
from RENlp_NER.config import config
from RENlp_NER.logger import logger

from RENlp_NER.models.Bert_NER import MyNER

app = Flask(__name__)
CORS(app)

model = MyNER("data/output")


@app.route("/predict", methods=['post'])
def predict():
    text = request.json["text"]
    # text = '海钓比赛的地点在厦门与金门之间的海域'
    try:
        result = model.predict(text)
        print(result)
        result = jsonify({'reslut': result})
        return result
    except Exception as e:
        print(e)
        return jsonify({"result": "Model Failed"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9001)
    predict()
