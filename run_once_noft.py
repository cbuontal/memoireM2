#
# Lance 1 entraînement sans fine-tuning
#

import argparse
from load_dataset import load_corpus
from tokenize_dataset import get_tokenized_dd, get_data_collator
from arguments import *
from metrics import *
import sys, os, csv
from transformers import AutoModelForTokenClassification, AutoTokenizer, EarlyStoppingCallback, Trainer
from optimisation_HP import get_optimal_parameters

def get_prf(trainer: Trainer):
    """
        extraire précision, rappel, f-mesure
    """
    metriques = trainer.evaluate()
    prfmesure = {}
    prfmesure['precision'] = round(metriques['eval_precision'], 3)
    prfmesure['rappel'] = round(metriques['eval_recall'], 3)
    prfmesure['fmesure'] = round(metriques['eval_f1'], 3)
    return prfmesure


parser = argparse.ArgumentParser(
    prog="run_once_noft",
    description="fait 1 run d'un modèle sur les 3 datasets et sauvegarde le fichier de métriques"
)

parser.add_argument('-m', '--model_path', help="path to huggingface model", required=True)
parser.add_argument('-t','--training_dataset', help='name of training dataset (txahl, chisiec, ACC)', choices=['txahl', 'chisiec', 'ACC'])
# si None -> tous les corpus
parser.add_argument('--cache_directory', help="cache directory where the model is saved", default='/data/cbuon/cache')
#parser.add_argument('--output_directory', help='directory in which checkpoints and models will be saved', required=True)

args = parser.parse_args()

# test si c'est un chemin HF
if len(str(args.model_path).split('/')) == 2:
    modelname = args.model_path.split('/')[0]
    destination = f"/data/cbuon/30_runs/{modelname}/{args.training_dataset}/"
else:
    modelname = args.model_path.split('/')[3]
    destination = f"/data/cbuon/30_runs/mes_modeles/{modelname}/{args.training_dataset}/"

#modelname = args.model_path.split('/')[0]

#print(args)

# 1. IMPORT DU DATASET

(dd, id2label, label2id) = load_corpus(args.training_dataset)
label_list = dd['train'].features[f"ner_tags"].feature.names

# 2. VECTORISATION DU DATASET

tokenized_dd = get_tokenized_dd(args.model_path, dd)
data_collator = get_data_collator(args.model_path)

# 3. MÉTRIQUES D'ÉVALUATION 

compute_metrics = get_compute_metrics(dd, label_list)

# 4. FINE-TUNING / ENTRAÎNEMENT 

# dossier où enregistrer le modèle
#parent_dir = "/data/cbuon/30_runs"
# du style /data/cbuon/30_runs/bert-chinese/txahl
#destination = f"{parent_dir}/{modelname}/{args.training_dataset}/"

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# pas de dossier spécifique à chaque run (il n'y en a qu'une)

#model_name_to_HF = {
#    'ethanyt': 'ethanyt/guwenbert-base',
#    'google-bert': 'google-bert/bert-base-chinese',
#    'hsc748NLP': 'hsc748NLP/GujiBERT_jian_fan',
#    'Jihuai': 'Jihuai/bert-ancient-chinese',
#    'KoichiYasuoka': 'KoichiYasuoka/roberta-classical-chinese-base-char',
#    'SIKU-BERT': 'SIKU-BERT/sikubert'
#}

model = AutoModelForTokenClassification.from_pretrained(
    args.model_path,
    num_labels=len(id2label.keys()), 
    id2label=id2label, 
    label2id=label2id,
    cache_dir = args.cache_directory
)

esc = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)

training_args = get_base()

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_dd["train"],
    eval_dataset=tokenized_dd["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[esc],
)

trainer.evaluate()

prf = get_prf(trainer)

# écriture du fichier csv contenant P, R, F1
# j'enregistre dans /data/cbuon/30_runs/<MODEL>/<CORPUS>/metrics_noft.csv

# test pour savoir si c'est un chemin HF
if len(str(args.model_path).split('/')) == 2:
    path = f"/data/cbuon/30_runs/{str(args.model_path).split('/')[0]}/{str(args.training_dataset)}/"
else:
    # c'est un modèle à moi
    path = f"/data/cbuon/30_runs/mes_modeles/{str(args.model_path).split('/')[3]}/{str(args.training_dataset)}/"

if os.path.exists(path) and os.path.isdir(path):
    pass
else:
    os.makedirs(path) # makedirs pour que ce soit récursif

metrics_file = path + "metrics_noft.csv"
with open(metrics_file, "w", newline="") as f:
    w = csv.DictWriter(f, prf.keys())
    w.writeheader()
    w.writerow(prf)