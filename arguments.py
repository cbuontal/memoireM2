from transformers import TrainingArguments, Trainer
from typing import Dict, List, Tuple

# pour l'étude ablative pour chercher les meilleurs
# hyper-paramètres

def get_args(output_dir: str, lr: float, batch_train: int, batch_eval: int, epochs: int, wd: float) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir, # là où sera sauvegardé le modèle
        logging_dir=output_dir + "/logs",
        logging_steps=10,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        learning_rate=lr, ###
        per_device_train_batch_size=batch_train, ##
        per_device_eval_batch_size=batch_eval, ##
        num_train_epochs=epochs,###
        weight_decay=wd, ###
        eval_strategy="epoch",
        save_strategy="epoch",
        #eval_steps = 10, # Evaluation and Save happens every 10 steps
        save_total_limit = 2, # https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442/8
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1', # par défaut 'loss' quand on ne précise
)

#### CONFIGURATION DE BASE

def get_base():
    return TrainingArguments(
    output_dir="/data/cbuon/jihuai_NER_juridique/outputs",
    logging_dir="/data/cbuon/jihuai_NER_juridique/logs",
    remove_unused_columns=False,
    learning_rate=2e-5, ###
    per_device_train_batch_size=32, ##
    per_device_eval_batch_size=16, ##
    num_train_epochs=20,###
    weight_decay=0.01, ###
    eval_strategy="epoch",
    save_strategy="epoch",
    #eval_steps = 10, # Evaluation and Save happens every 10 steps
    save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
)

def get_arguments_dict(output_dir='/tmp/optimisation', batch_size_training=16):
    """

    """
    #### TEST DU LEARNING RATE
    
    LR = []
    test_LR = []
    learning_rates = [2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4]
    for lr in learning_rates:
        test_LR.append((lr, get_args(output_dir, lr, batch_size_training, 16, 10, 0.01)))
    
    #### TEST DU WEIGHT DECAY
    
    test_WD = []
    weight_decays = [0.001, 0.005, 0.008, 0.05, 0.08, 0.1]
    for wd in weight_decays:
        test_WD.append((wd, get_args(output_dir, 2e-5, batch_size_training, 16, 10, wd)))
    
    # chaque élément est une liste de tuples (valeur, training_args)
    TRAINING_ARGUMENTS = {
        "learning_rate": test_LR,
        "weight_decay": test_WD
    }

    return TRAINING_ARGUMENTS