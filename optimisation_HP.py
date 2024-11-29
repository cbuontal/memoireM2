#
# Estime les hyperparamètres optimaux pour une combinaison
# modèle / corpus
#
# on regarde :
# - learning rate
# - weight decay
#

from load_dataset import load_corpus
from tokenize_dataset import get_tokenized_dd, get_data_collator
from metrics import *
from arguments import *
from transformers import AutoModelForTokenClassification, AutoTokenizer
import argparse, sys, os, csv, glob
import polars as pl

def get_optimal_parameters(modelname: str, corpus: str):
    """
        En fonction du modèle et du corpus, lit le fichier des parametres optimaux et
        renvoie le tupe (lr, wd) optimal.
    """
    if modelname.startswith('my_'):
        path = f"/data/cbuon/optimisation/mes_modeles/{modelname}/{corpus}/parametres.csv"
    else:
        path = f"/data/cbuon/optimisation/{modelname}/{corpus}/parametres.csv"
    optim = pl.read_csv(path).with_columns(pl.col("fmesure").max().over("parametre").alias("max_fmesure"))
    result = optim.filter(pl.col("fmesure") == pl.col("max_fmesure"))
    result = result.drop("max_fmesure")
    # learning rate, weight decay
    return (list(result['valeur'])[0], list(result['valeur'])[1])

def main():
    parser = argparse.ArgumentParser(
        prog="optimisation_hyperparamètres",
        description="recherche les meilleurs hyper paramètres pour une combinaison de modèle / corpus"
    )
    
    parser.add_argument('-m', '--model_path', help="path to huggingface model", required=True)
    parser.add_argument('-t', '--training_dataset', help='name of training dataset (txahl, chisiec, ACC, ACC_NER)', choices=['txahl', 'chisiec', 'ACC', 'ACC_NER'], required=True)
    parser.add_argument('--log_dir', help='path to logging directory')
    parser.add_argument('--batch_size_training', help='size of training batches', default=16)
    parser.add_argument('--cache_directory', help="cache directory where the model is saved", default='/data/cbuon/cache')
    
    args = parser.parse_args()
    # test si c'est un chemin HF
    if len(str(args.model_path).split('/')) == 2:
        modelname = args.model_path.split('/')[0]
        pathname = f"/data/cbuon/optimisation/{modelname}/{args.training_dataset}/"
    else:
        modelname = args.model_path.split('/')[3]
        pathname = glob.glob(f"/data/cbuon/optimisation/mes_modeles/{modelname}/{args.training_dataset}/")

    if os.path.exists(pathname) and os.path.isdir(pathname):
        pass
    else:
        os.makedirs(pathname) # makedirs pour que ce soit récursif
    
    # log_dir par défaut
    if args.log_dir is None:
        log_dir = pathname + "logs/"
    else:
        log_dir = args.log_dir
    
    os.makedirs(log_dir, exist_ok = True)
    
    # 1. IMPORT DU DATASET
    
    (dd, id2label, label2id) = load_corpus(args.training_dataset)
    
    # 2. VECTORISATION DU DATASET
    
    tokenized_dd = get_tokenized_dd(args.model_path, dd)
    data_collator = get_data_collator(args.model_path)
    
    # 3. TEST DES HYPERPARAMÈTRES
    
    perfs = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    arguments_dict = get_arguments_dict(pathname, int(args.batch_size_training))
    
    label_list = dd['train'].features[f"ner_tags"].feature.names
    compute_metrics = get_compute_metrics(dd, label_list)
    
    for variable, trainingargs in arguments_dict.items():
        # pour chaque variable à tester
        for trainingarg in trainingargs:
            # pour chaque valeur de la variable (valeur, TArgs)
            # recharger le modèle pour repartir de 0 à l'entraînement
    
            model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, 
            num_labels=len(id2label.keys()), 
            id2label=id2label,
            label2id=label2id,
            cache_dir = "/data/cbuon/cache"
            )
            
            trainer = Trainer(
                model=model,
                args=trainingarg[1],
                train_dataset=tokenized_dd["train"],
                eval_dataset=tokenized_dd["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            trainer.train()
    
            # une fois que c'est entraîné j'enregistre les mesures 
            prf = get_prf(trainer)
            perfs.append((variable, trainingarg[0], prf['precision'], prf['rappel'], prf['fmesure']))
    
    schema = {
        'parametre': pl.String,
        'valeur': pl.Float32,
        'precision': pl.Float32,
        'rappel': pl.Float32,
        'fmesure': pl.Float32
    }
    
    perfs = pl.DataFrame(perfs, schema=schema)
    
    path = pathname + "parametres.csv"
    perfs.write_csv(path, separator=",")

if __name__ == '__main__':
    main()