import os
from pathlib import Path

ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

# data
data_dir = os.path.join(ROOT_PATH, "data")
model_dir = os.path.join(ROOT_PATH, "model")

bert_data_path = Path.home() / '.pytorch_pretrained_bert'
bert_vocab_path = bert_data_path / 'bert-base-chinese-vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese'

tencent_w2v_path = Path.home() / '.word2vec'

roberta_model_path = bert_data_path / 'chinese_Roberta_bert_wwm_large_ext_pytorch'

common_data_path = Path.home() / '.common_dataset'
intent_data_path = Path(common_data_path) / 'intent_data'

intent_labels = ['保全', '保障范围_保费费率_其他', '保障范围_保费费率_属性', '保障范围_保险条款', '保障范围_保险责任',
                 '保障范围_保额_其他', '保障范围_保额_属性', '保障范围_免赔额_其他', '保障范围_免赔额_属性',
                 '保障范围_免赔额_概念', '保障范围_医院_其他', '保障范围_医院_范围', '保障范围_医院垫付',
                 '保障范围_外购药_其他', '保障范围_外购药_属性', '保障范围_法律援助_其他', '保障范围_法律援助_概念',
                 '保障范围_特殊门诊', '保障范围_等待期_其他', '保障范围_等待期_属性', '保障范围_等待期_概念', '保障范围_缴费频率',
                 '保障范围_门急诊', '保障范围_门诊手术', '投保', '查询_保单', '查询_公司', '核保', '核保_不可投保疾病',
                 '核保_不可投保职业', '核保_保前体检_其他', '核保_保前体检_属性', '核保_健康告知', '核保_其他', '核保_可投保人',
                 '核保_投保区域', '核保_投保证件', '核保_被保险人年龄', '核保_被保险人年龄_不可投保疾病', '核赔', '理赔',
                 '理赔_理赔流程', '社保_属性', '社保_核保', '社保_核赔', '续保', '续保_可连续投保年龄', '续保_理赔后续保',
                 '续保_续保保费', '负样本', '退保']

