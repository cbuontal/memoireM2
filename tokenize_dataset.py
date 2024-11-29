#
# À partir du nom du modèle, du dataset_dict et de la liste des labels,
# tokenize le dataset_dict et renvoie un tokenized_dd
#
#

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

def get_data_collator(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return DataCollatorForTokenClassification(tokenizer=tokenizer)

def get_tokenized_dd(tokenizer_path, dd) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenized_dd = dd.map(tokenize_and_align_labels, batched=True, fn_kwargs={'tokenizer' : tokenizer})
    # on enlève les colonnes inutiles
    tokenized_dd = tokenized_dd.remove_columns('token_type_ids')
    tokenized_dd = tokenized_dd.remove_columns('tokens')
    tokenized_dd = tokenized_dd.remove_columns('ner_tags')
    tokenized_dd = tokenized_dd.remove_columns('id')
    return tokenized_dd

def tokenize_and_align_labels(examples, tokenizer):
    # avec padding
    tokenized_inputs = tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True, max_length=512)

    labels = []
    # on parcourt une liste de listes contenant les tags NER de chaque entrée du dataset
    for i, label in enumerate(examples[f"ner_tags"]):
        # Map tokens to their respective word
        # '.word_ids()' returns a list mapping the tokens to their actual word
        # in the initial sentence for a fast tokenizer.
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        
        # on se prépare à itérer
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Set the special tokens to -100
            # 1 mot = 1 label qui porte sur le 1er token
            if word_idx is None:
                # pas d'indice -> ça a été rajouté par le tokenizer (sep, cls…)
                # Only label the first token of a given word.
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  
                # nouveau mot
                label_ids.append(label[word_idx])
            else:
                # si c'est pas un nouveau mot
                label_ids.append(-100)
            # on passe au mot suivant
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
    