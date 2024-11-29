#
# Le but de ce script est de charger les données demandées (modèle, dataset...)
# et de lancer les entraînements avec le bon comportement (callbacks, hyperparamètres...)
# en ligne de commande.
#

import argparse
from load_dataset import load_corpus
from tokenize_dataset import get_tokenized_dd, get_data_collator
from arguments import *
from metrics import *
import sys, os, csv
from transformers import AutoModelForTokenClassification, AutoTokenizer, EarlyStoppingCallback
from optimisation_HP import get_optimal_parameters

parser = argparse.ArgumentParser(
    prog="get_30_runs",
    description="crée 30 runs pour une combinaison modèle/dataset"
)

parser.add_argument('-m', '--model_path', help="path to huggingface model", required=True)
parser.add_argument('-t','--training_dataset', help='name of training dataset (txahl, chisiec, ACC, ACC_NER)', choices=['txahl', 'chisiec', 'ACC', 'ACC_NER'], required=True)
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

if os.path.exists(destination) and os.path.isdir(destination):
    pass
else:
    os.makedirs(destination) # makedirs pour que ce soit récursif

# 1. IMPORT DU DATASET

(dd, id2label, label2id) = load_corpus(args.training_dataset)
label_list = dd['train'].features[f"ner_tags"].feature.names

# 2. VECTORISATION DU DATASET

tokenized_dd = get_tokenized_dd(args.model_path, dd)
data_collator = get_data_collator(args.model_path)

# 3. MÉTRIQUES D'ÉVALUATION 

compute_metrics = get_compute_metrics(dd, label_list)

# 4. FINE-TUNING / ENTRAÎNEMENT 

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

for i in range(30):
    # écriture du dossier spécifique à la run
    directory = "run_" + str(i+1) + "/"
    path = os.path.join(destination, directory)
    if os.path.exists(path) and os.path.isdir(path):
        pass
    else:
        os.makedirs(path) # makedirs pour que ce soit récursif
    
    # reset le modèle
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=len(id2label.keys()), 
        id2label=id2label, 
        label2id=label2id,
        cache_dir = "/data/cbuon/cache"
    )

    esc = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)

    # wd, lr optimaux selon le modèle
    (lr, wd) = get_optimal_parameters(modelname, args.training_dataset)
    
    trainingarg = get_args(str(path), lr, 32, 16, 30, wd)
    
    trainer = Trainer(
        #output_dir = str(path),
        model=model,
        args=trainingarg,
        train_dataset=tokenized_dd["train"],
        eval_dataset=tokenized_dd["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[esc],
    )
    
    trainer.train()
    
    # une fois que c'est entraîné j'enregistre les mesures 
    prf = get_prf(trainer)
    
    # écriture du fichier csv contenant P, R, F1
    metrics_file = path + "metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        w = csv.DictWriter(f, prf.keys())
        w.writeheader()
        w.writerow(prf)