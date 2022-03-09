from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch.nn as nn

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def loss_fn(outputs, targets):
    print(outputs)
    print(targets)
    #return nn.CrossEntropyLoss()(outputs, targets)
    return nn.MSELoss(outputs, targets)
    #return torch.nn.BCEWithLogitsLoss()(outputs,targets)

def truncate_left_text_dataset(dataframe, tokenizer):
    #if we want to only look at the last 512 tokens of a dataset

    for i,row in dataframe.iterrows():
        tokens = tokenizer.tokenize(row['text'])
        strings = tokenizer.convert_tokens_to_string( ( tokens[-512:] ) )
        dataframe.loc[i, 'text'] = strings

    return dataframe
