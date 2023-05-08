import argparse
import os
import torch
from transformers import AutoTokenizer
from transformers import BertModel
from utils import set_seed, load_data
from train_and_evaluate import train_cp, evaluate_cp
from covid_dataset import Covid19StanceDatasetCP
from stance_model import SModel

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--warmup", type=float, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ratio", type=int, required=False, default=3)
parser.add_argument("--model_name", type=str, default='ct-bert.bin')
parser.add_argument("--save_path", type=str, default='./')
args = parser.parse_args()

'''
LR=(1e-6 4e-6 7e-6 9e-6 2e-5 5e-5)
WARM_UP=(0.0 0.1 0.4)
EPOCHS=(3 5 7)
'''

def main():
    set_seed(42)
    #file_path = "~/runs/lr=" + str(args.lr).replace("e-0", "e-") + "/warmup=" + str(args.warmup) + "/epochs=" + str(args.epochs) + "/result.txt"
    # load models
    if args.model == 'CTBERT':
        model_type = 'digitalepidemiologylab/covid-twitter-bert-v2'
        if not os.path.exists('covid-bert-model'):
            os.mkdir('covid-bert-model')
            os.system('wget https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2/resolve/main/pytorch_model.bin -P covid-bert-model/')
            os.system(
                'wget https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2/resolve/main/config.json -P covid-bert-model/')
            os.system(
                'wget https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2/resolve/main/tokenizer_config.json -P covid-bert-model/')
            os.system(
                'wget https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2/resolve/main/vocab.txt -P covid-bert-model/')
            os.system(
                'wget https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2/resolve/main/special_tokens_map.json -P covid-bert-model/')
    else:
        model_type = 'bert-base-uncased'

    encoder = BertModel.from_pretrained(model_type,
                                        output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    cp_marker = ['<cp_start>', '<cp_end>']
    tokenizer.add_tokens(cp_marker)
    encoder.resize_token_embeddings(len(tokenizer))

    print('Initializing model...')
    smodel = SModel(encoder, 3)
    smodel.cuda()

    # load data
    texts, labels, cp = load_data('..\covid_19_treatment_stance_dataset.csv')
    train_dataset_cp, dev_dataset_cp, test_dataset_cp, train_label, dev_label, test_label = Covid19StanceDatasetCP.get_processed_data(texts, labels, cp, tokenizer, args.ratio * 0.05 + 0.45)

    print('Training...')
    best_model, best_f1 = train_cp(train_dataset_cp, dev_dataset_cp, smodel, lr=args.lr, warmup_ratio=args.warmup, num_epochs=args.epochs)

    print(args.lr, args.warmup, args.epochs)
    print("F1_FINAL:", best_f1)
    
    torch.save(best_model.state_dict(), args.save_path)
    
    dev_f1 = evaluate_cp(best_model, dev_dataset_cp)
    test_f1 = evaluate_cp(best_model, test_dataset_cp)

    print('DEV {}, TEST {}'.format(dev_f1, test_f1))


if __name__ == "__main__":
    main()