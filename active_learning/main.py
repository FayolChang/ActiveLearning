from active_learning.infer import infer_main
from active_learning.prepare_data import select_samples
from active_learning.train import train_main

# for p in [20, 40, 60, 80]:
for p in [20]:
    train_main(p)
    infer_main(100-p)
    select_samples(p, 100-p, p+20, 80-p)

train_main('train_data.json')