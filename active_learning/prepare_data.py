import collections
import json
import random
from pathlib import Path
import numpy
from configuration.config import intent_data_path, data_dir, logger


def split_data():
    train = json.load(Path(intent_data_path/'train_data.json').open())
    print(f'train size: {len(train)}')  # 28686

    labels = collections.defaultdict(list)
    for d in train:
        labels[d['label']].append(d['text'])

    print(f'labels are: ')
    print(sorted(labels.keys()))

    d_20, d_80 = [], []
    for k, vs in labels.items():
        vs_cnt = len(vs)
        selected = random.sample(vs, vs_cnt//5)
        for _ in selected:
            d_20.append({
                'label': k,
                'text': _
            })
        for _ in vs:
            if _ in selected: continue
            d_80.append({
                'label': k,
                'text': _
            })
    print(f'd_20 size: {len(d_20)}')  # d_20 size: 5716 d_80 size: 22963
    print(f'd_80 size: {len(d_80)}')

    json.dump(d_20, (Path(data_dir)/'data_20_per.json').open('w'), ensure_ascii=False, indent=2)
    json.dump(d_80, (Path(data_dir)/'data_80_per.json').open('w'), ensure_ascii=False, indent=2)


def a_d80():
    d_80 = json.load((Path(data_dir)/'d_80_06011144.json').open())
    # 分析指标：max、min、avg、median
    # 1.总 []
    # 2.每个类别 {label: []}
    # 3.预测正确 总 []
    # 4.预测正确 每个类别 {label: []}
    # 5.预测错误 总 []
    # 6.预测错误 每个类别 {label: []}
    total, right, wrong = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)

    for d in d_80:
        true_label, pred_label, text, score = d['true_label'], d['pred_label'], d['text'], float(d['score'])
        total[true_label].append(score)
        if true_label == pred_label:
            right[true_label].append(score)
        else:
            wrong[true_label].append(score)
    
    total_scores = [_ for l in total.values() for _ in l]
    print(f'总 - max: {max(total_scores)} - min: {min(total_scores)} - avg: {numpy.mean(total_scores)} '
          f'- median: {numpy.median(total_scores)} - percentile: {numpy.percentile(total_scores, [25])[0]}')
    print('\n')
    
    for k, vs in total.items():
        print(f'total {k} - max: {max(vs)} - min: {min(vs)} - avg: {numpy.mean(vs)} '
              f'- median: {numpy.median(vs)} - percentile: {numpy.percentile(vs, [25])[0]}')
    print('\n')

    total_scores = [_ for l in right.values() for _ in l]
    print(f'right - max: {max(total_scores)} - min: {min(total_scores)} - avg: {numpy.mean(total_scores)} '
          f'- median: {numpy.median(total_scores)} - percentile: {numpy.percentile(total_scores, [25])[0]}')
    print('\n')

    for k, vs in right.items():
        print(f'right {k} - max: {max(vs)} - min: {min(vs)} - avg: {numpy.mean(vs)} '
              f'- median: {numpy.median(vs)} - percentile: {numpy.percentile(vs, [25])[0]}')
    print('\n')

    total_scores = [_ for l in wrong.values() for _ in l]
    print(f'wrong - max: {max(total_scores)} - min: {min(total_scores)} - avg: {numpy.mean(total_scores)} '
          f'- median: {numpy.median(total_scores)} - percentile: {numpy.percentile(total_scores, [25])[0]}')
    print('\n')

    for k, vs in wrong.items():
        print(f'{k} - max: {max(vs)} - min: {min(vs)} - avg: {numpy.mean(vs)} '
              f'- median: {numpy.median(vs)} - percentile: {numpy.percentile(vs, [25])[0]}')
    print('\n')


def select_samples(labeled_file, unlabeled_file, o_labeled_file, o_unlabeled_file):
    labeled_file = f'labeled_{labeled_file}'
    unlabeled_file = f'unlabeled_{unlabeled_file}'
    o_labeled_file = f'labeled_{o_labeled_file}'
    o_unlabeled_file = f'unlabeled_{o_unlabeled_file}'

    low_cnt, random_cnt = 5400, 340

    d_unlabeled = json.load((Path(data_dir) / unlabeled_file).open())
    d_labeled = json.load((Path(data_dir) / labeled_file).open())

    d_unlabeled = sorted(d_unlabeled, key=lambda x: float(x['score']))

    low_samples, d_unlabeled = d_unlabeled[:low_cnt], d_unlabeled[low_cnt:]
    random_samples = random.sample(d_unlabeled, random_cnt)
    for _ in random_samples:
        d_unlabeled.remove(_)

    d_to_add = low_samples + random_samples
    d_as_unlabeled = [_ for _ in d_unlabeled if _ not in d_to_add]

    def trans(data):
        for dic in data:
            dic['label'] = dic['pred_label']
            dic.pop('pred_label')
            dic.pop('true_label')
            dic.pop('score')

    trans(d_to_add)
    trans(d_as_unlabeled)

    json.dump(d_to_add + d_labeled, (Path(data_dir)/o_labeled_file).open('w'), ensure_ascii=False, indent=2)
    json.dump(d_as_unlabeled, (Path(data_dir)/o_unlabeled_file).open('w'), ensure_ascii=False, indent=2)

    logger.info(f'input: {labeled_file} - {unlabeled_file}')
    logger.info(f'd_to_add size: {len(d_to_add)}')
    logger.info(f'd_as_unlabeled size: {len(d_as_unlabeled)}')
    logger.info(f'd_to_add + d_labeled size: {len(d_to_add) + len(d_labeled)}')

    # new_d_20 size: 5740
    # left_60 size: 17223
    # pre_40 size: 11456

    # new_d_20 size: 5741
    # left_40 size: 11482
    # pre_60 size: 17197

    # d_to_add size: 5741
    # d_as_unlabeled size: 5741
    # d_to_add + d_labeled size: 22938


# select_40('data_20_per.json', 'd_80_06011144.json', 4, 'data_40_per.json', 'data_60_per.json')
# select_samples('data_40_per.json', 'd_60_06011626.json', 3, 'data_60_per_l.json', 'data_40_per_u.json')
# select_samples('data_60_per_l.json', 'd_40_06011749.json', 2, 'data_80_per_l.json', 'data_20_per_u.json')





