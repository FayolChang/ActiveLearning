from active_learning import train, infer, prepare_data
from active_learning.train import train_main

for p in [20, 40, 60, 80]:
    train.train_main(p)
    infer.infer_main(100-p)
    prepare_data.select_samples(p, 100-p, p+20, 80-p)

train_main('train_data.json')