# import des données pour l'entraînement de jihuai sur du fine-tuning


from datasets import Dataset, DatasetDict, ClassLabel, Features, Value, Sequence
from typing import Dict, List, Tuple
import json

# PATH
path_to_data = "tagging_xahl/Data/"

def get_train():
    traindata = path_to_data + "corpus_train.jsonl"
    with open(traindata, 'r') as f:
        jsonl_content = f.read()
    # liste où 1 item = 1 dictionnaire
    data = [json.loads(jline) for jline in jsonl_content.splitlines()]
    # enlever les dernières catégories :
    for entry in data:
        del(entry["meta"])
        del(entry["answer"])
        del(entry["_session_id"])
    return data

def get_test():
    testdata = path_to_data + "corpus_test.jsonl"
    with open(testdata, 'r') as f:
        jsonl_content = f.read()
    # liste où 1 item = 1 dictionnaire
    data = [json.loads(jline) for jline in jsonl_content.splitlines()]
    # enlever les dernières catégories :
    for entry in data:
        del(entry["meta"])
        del(entry["answer"])
        del(entry["_session_id"])
    return data

def get_eval():
    testdata = path_to_data + "corpus_eval.jsonl"
    with open(testdata, 'r') as f:
        jsonl_content = f.read()
    # liste où 1 item = 1 dictionnaire
    data = [json.loads(jline) for jline in jsonl_content.splitlines()]
    # enlever les dernières catégories :
    for entry in data:
        del(entry["meta"])
        del(entry["answer"])
        del(entry["_session_id"])
    return data

def get_labels(data: List[Dict]) -> List:
    labels = list()
    for d in data:
        for elt in d['spans']:
            if elt["label"] not in labels:
                labels.append(elt["label"])
    return labels

def get_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    """
        Renvoie un tuple (train_dataset, test_dataset, eval_dataset)
        d'objets Dataset avec le format :

        Dataset({
            features: ['text', 'spans'],
            num_rows: 309
        })
    """
    train = get_train()
    test = get_test()
    evals = get_eval()

    labels_train = set(get_labels(train))
    labels_test = set(get_labels(test))
    labels_eval = set(get_labels(evals))
    labels = list(labels_train | labels_test | labels_eval) # on merge les listes

    # définition des tags NER :
    tags = ['O']
    for lab in labels:
        tags.append("B-" + lab)
        tags.append("I-" + lab)
    
    # Définir les features du Dataset
    features = Features({
        'text': Value('string'),
        'spans': Sequence({
            'label': ClassLabel(names=labels),
            'start': Value('int32'),
            'end': Value('int32'),
            'token_start': Value('int32'),
            'token_end': Value('int32')
        })
    })

    train_dataset = Dataset.from_list(train, features=features)
    test_dataset = Dataset.from_list(test, features=features)
    eval_dataset = Dataset.from_list(evals, features=features)
    return (train_dataset, test_dataset, eval_dataset)

def get_goal_set(dataset) -> List[Dict]:
    """
        Prend en entrée train, test ou eval
        
        Produit en sortie une liste, contenant des dictionnaires
        avec les features voulues :
            'id',
            'tokens',
            'ner_tags'
            
        + trie les entrées sans annotation ('spans' vide)
    """
    goal = []
    for idx, sample in enumerate(dataset):
        # il faut vérifier que l'entrée est non vide
        if sample['spans'] == []:
            continue
        dico = {}
        dico['id'] = idx
        tokens = [c for c in sample['text']]
        # il faut vérifier selon le max du tokenizer
        if len(tokens) < 512: # (512)
            # ok on enregistre
            dico['tokens'] = tokens
            # liste de NER tags de la même longueur
            ner_tags = ['O'] * len(tokens)
            for entry in sample['spans']:
                start = entry['start']
                end = entry['end']
                ner_tags[start] = 'B-' + entry['label']
                for i in range(1, end-start):
                    ner_tags[start+i] = 'I-' + entry['label']
            dico['ner_tags'] = ner_tags
            # on ajoute à la liste
            goal.append(dico)
        else:
            continue
            # ça dépasse la limite donc il faut couper !
            # nombre de coupes de 511 sinogrammes + 1 s'il reste encore des tokens
            nb_coupes = len(tokens) // 511 + int(len(tokens) % 511 != 0)
            #for i in range(nb_coupes)
    return goal

def get_all_tags(train, test, evals):
    tags_entites = set()
    for corpus in [get_goal_set(train), get_goal_set(test), get_goal_set(evals)]:
        for dico in corpus:
            tags_entites = tags_entites | set(dico['ner_tags'])
    return list(tags_entites)

