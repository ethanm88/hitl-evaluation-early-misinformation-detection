import csv
import random
import torch
import numpy as np

LABEL_TO_INDEX = {'Agree': 0, 'Disagree': 1, 'No Stance': 2}

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def load_data(data_dir):
    texts = []
    labels = []
    cure_prevention = []
    with open(data_dir, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx != 0:
                texts.append(row[1])
                labels.append(LABEL_TO_INDEX[row[-1]])
                cure_prevention.append(row[-2])

    return texts, labels, cure_prevention