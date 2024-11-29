from transformers import Trainer
import evaluate
import numpy as np

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

def get_compute_metrics(dd, label_list):
    def compute_metrics(p):
        seqeval = evaluate.load("seqeval")
        labels = list(set([label_list[i] for i in dd["train"][0][f"ner_tags"]]))
        predictions, labels = p
        # sélection de la prédiction la plus probable
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics