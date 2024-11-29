#
# Renvoie le tuple (train, test, eval) pour le corpus
# demandé
#
#

from format_data import *
import sys
import csv

def load_corpus(corpus: str) -> Tuple[DatasetDict, Dict, Dict]:
    """
        Renvoie le tuple (dataset_dict, liste_labels) selon
        le corpus choisi
    """
    if corpus not in ['chisiec', 'txahl', 'ACC', 'ACC_NER']:
        sys.exit(1)
    if corpus == 'chisiec':
        return get_chisiec()
    elif corpus == 'txahl':
        return get_txahl()
    elif corpus == 'ACC':
        return get_ACC()
    elif corpus == 'ACC_NER':
        return get_ACC_NER()

def get_txahl() -> Tuple[Dataset, Dict, Dict]:
    """
        
    """
    (train, test, evals) = (get_train(), get_test(), get_eval())
    # mêmes features que le tuto HF
    ner_features = Features({
        'id': Value(dtype='int32'),
        'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'ner_tags': Sequence(feature=ClassLabel(names=get_all_tags(train, test, evals), id=None), length=-1, id=None)
    })
    
    # je ne garde que les exemples de moins de 512 tokens parce que
    # c'est la limite du tokenizer.
    goal_train = Dataset.from_list(get_goal_set(train), features=ner_features).filter(lambda x: len(x['tokens']) < 512)
    goal_test = Dataset.from_list(get_goal_set(test), features=ner_features).filter(lambda x: len(x['tokens']) < 512)
    goal_eval = Dataset.from_list(get_goal_set(evals), features=ner_features).filter(lambda x: len(x['tokens']) < 512)
    
    dd_txahl = DatasetDict({"train":goal_train, "test":goal_test, 'evaluation':goal_eval})
    label_list_txahl = dd_txahl['train'].features[f"ner_tags"].feature.names

    label2id = {}
    id2label = {}
    for idx, tag in enumerate(label_list_txahl):
        label2id[tag] = idx
        id2label[idx] = tag

    return (dd_txahl, id2label, label2id)

def get_chisiec() -> Tuple[Dataset, Dict, Dict]:
    """
        prend le .csv au format BIO et renvoie le datasetdict
        et les id2label / label2id
    """
    dev = get_corpus("CHisIEC/data/ner/dev_bio2.txt")
    test = get_corpus("CHisIEC/data/ner/test_bio2.txt")
    train = get_corpus("CHisIEC/data/ner/train_bio2.txt")

    # récupérer les tags
    tags = set()
    for corpus in [dev, train]:
        for elt in corpus:
            tags = tags | set(elt['ner_tags']) 
    id2label = {}
    label2id = {}
    for idx, tag in enumerate(list(tags)):
        id2label[idx] = tag
        label2id[tag] = idx
    
    ner_features = Features({
        'id': Value(dtype='int32'),
        'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'ner_tags': Sequence(feature=ClassLabel(names=list(tags), id=None), length=-1, id=None)
    })
    
    goal_dev = Dataset.from_list(dev, features=ner_features).filter(lambda x: len(x['tokens']) < 512) 
    goal_train = Dataset.from_list(train, features=ner_features).filter(lambda x: len(x['tokens']) < 512) 
    goal_test = Dataset.from_list(test, features=ner_features).filter(lambda x: len(x['tokens']) < 512) 

    dd_chisiec = DatasetDict({"train":goal_train, "test":goal_test, 'eval':goal_dev})
    label_list_chisiec = dd_chisiec['train'].features[f"ner_tags"].feature.names

    return (dd_chisiec, id2label, label2id)

def get_ACC() -> Tuple[Dataset, Dict, Dict]:
    """
        prend le .tsv au format BIO et renvoie le datasetdict
        et les id2label / label2id
    """
    dev = get_corpus("/data/cbuon/datasets/ACC/dev.tsv")
    assert(dev != [])
    test = get_corpus("/data/cbuon/datasets/ACC/test.tsv")
    assert(test != [])
    train = get_corpus("/data/cbuon/datasets/ACC/train.tsv")
    assert(train != [])

    # récupérer les tags
    tags = set()
    for corpus in [dev, train, test]:
        for elt in corpus:
            tags = tags | set(elt['ner_tags']) 
    id2label = {}
    label2id = {}
    for idx, tag in enumerate(list(tags)):
        id2label[idx] = tag
        label2id[tag] = idx
    
    ner_features = Features({
        'id': Value(dtype='int32'),
        'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'ner_tags': Sequence(feature=ClassLabel(names=list(tags), id=None), length=-1, id=None)
    })
    
    goal_dev = Dataset.from_list(dev, features=ner_features).filter(lambda x: len(x['tokens']) < 510) 
    goal_train = Dataset.from_list(train, features=ner_features).filter(lambda x: len(x['tokens']) < 510) 
    goal_test = Dataset.from_list(test, features=ner_features).filter(lambda x: len(x['tokens']) < 510) 

    dd_ACC = DatasetDict({"train":goal_train, "test":goal_test, 'eval':goal_dev})
    label_list_ACC = dd_ACC['train'].features[f"ner_tags"].feature.names

    return (dd_ACC, id2label, label2id)

def get_ACC_NER() -> Tuple[Dataset, Dict, Dict]:
    """
        corpus ACC mais en NER : seulement les ner-tags.
        
        prend le .tsv au format BIO et renvoie le datasetdict
        et les id2label / label2id
    """
    dev = get_corpus("/data/cbuon/datasets/ACC_NER/dev.tsv")
    assert(dev != [])
    test = get_corpus("/data/cbuon/datasets/ACC_NER/test.tsv")
    assert(test != [])
    train = get_corpus("/data/cbuon/datasets/ACC_NER/train.tsv")
    assert(train != [])

    # récupérer les tags
    tags = set()
    for corpus in [dev, train, test]:
        for elt in corpus:
            tags = tags | set(elt['ner_tags']) 
    id2label = {}
    label2id = {}
    for idx, tag in enumerate(list(tags)):
        id2label[idx] = tag
        label2id[tag] = idx
    
    ner_features = Features({
        'id': Value(dtype='int32'),
        'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'ner_tags': Sequence(feature=ClassLabel(names=list(tags), id=None), length=-1, id=None)
    })
    
    goal_dev = Dataset.from_list(dev, features=ner_features).filter(lambda x: len(x['tokens']) < 510) 
    goal_train = Dataset.from_list(train, features=ner_features).filter(lambda x: len(x['tokens']) < 510) 
    goal_test = Dataset.from_list(test, features=ner_features).filter(lambda x: len(x['tokens']) < 510) 

    dd_ACC = DatasetDict({"train":goal_train, "test":goal_test, 'eval':goal_dev})
    label_list_ACC = dd_ACC['train'].features[f"ner_tags"].feature.names

    return (dd_ACC, id2label, label2id)


def get_corpus(filename: str) -> list:
    """
        À partir du nom de fichier, renvoie la liste des éléments,
        chacun sous forme de dictionnaire, avec :
        - id (int)
        - tokens (liste)
        - ner_tags (liste)
    """
    corpus = list()
    # parcours d'un fichier formaté en 
    # token    \t    ner_tag 
    # les exemples sont séparés par des lignes vides 
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        identifiant = 0
        # exemple courant
        current_element = {}
        current_element['id'] = identifiant
        current_element['tokens'] = list()
        current_element['ner_tags'] = list()
        for row in reader:
            if row != []:
                # on ajoute au dictionnaire courant
                current_element['tokens'].append(row[0])
                current_element['ner_tags'].append(row[1])
            else:
                # ligne vide -> on enregistre le dictionnaire courant
                # COPIE PROFONDE en utilisant le constructeur de classe
                corpus.append(dict(current_element))
                # on passe à l'exemple suivant
                identifiant += 1
                current_element['id'] = identifiant
                current_element['tokens'] = list() # reset
                current_element['ner_tags'] = list() # reset
    return corpus