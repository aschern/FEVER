import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ast
from catboost import CatBoostClassifier, Pool
import json
import logging
import numpy as np
import pandas as pd
import pickle
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm_notebook as tqdm
from unidecode import unidecode


logging.getLogger().setLevel(logging.INFO)


data_path = os.path.dirname(os.path.abspath(''))
wiki_path = os.path.join(data_path, "FEVER_data/wiki_pages.csv")
answers = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']    # used to replace numbers and corresponding values
w_decoded = {}


def load_wiki(path="FEVER_data/wiki_pages.csv"):
    global w_decoded
    logging.info('loading Wikipedia dump')
    wiki = pd.read_csv(path)  #nrows=70000
    w_decoded = dict()
    for r in wiki.iterrows():
        if r[1]['id'] == r[1]['id']:
            w_decoded[unidecode(r[1]['id'])] = r[1]['id']
    logging.info('done')


def get_prediction_format(claims, pred, bert_res, k=20):
    '''
    Args:
    claims
    pred - data from a file passed for prediction in the model (contains fields text_a, text_b, sent, title, claim index)
    bert_res - bert prediction
    k - the number of evidence sentences for each claim (20 as default)
    
    Returns:
    one row of length k for each claim: combined data from pred and bert_res (with padding)
    '''
    logging.info('prepare data')
    comb = pred.join(bert_res)
    comb['SUPPORTS'] = comb['SUPPORTS'].fillna(0.0)
    comb['REFUTES'] = comb['REFUTES'].fillna(0.0)
    comb['NOT ENOUGH INFO'] = comb['NOT ENOUGH INFO'].fillna(1.0)

    predictions = []
    for i in range(len(claims)):
        prediction = list(comb.loc[comb['index'] == i][answers].values.flatten())
        while len(prediction) < k * 3:
            prediction += [0.0, 0.0, 1.0]    # padding with NEI prediction
        predictions.append(prediction[:k * 3])
    predictions = np.array(predictions)
    logging.info('done')
    return predictions


def logical_aggregation(predictions, k=20):
    preds = []
    for prediction in predictions:
        scores = []
        ans = []
        for i in range(k):
            ans.append(answers[prediction[i * 3: (i + 1) * 3].argmax()])
            scores.append(np.max(prediction[i * 3: (i + 1) * 3]))
        
        s = ans.count('SUPPORTS')
        r = ans.count('REFUTES')
        if s == 0 and r == 0:
            preds.append('NOT ENOUGH INFO')
        elif s == 0:
            preds.append('REFUTES')
        elif r == 0:
            preds.append('SUPPORTS')
        else:
            l = ''
            cur_max = 0
            for i in range(len(ans)):
                if ans[i] != 'NOT ENOUGH INFO' and abs(scores[i]) > cur_max:
                    l = ans[i]
                    cur_max = abs(scores[i])
            preds.append(l)
    return np.array(preds)


def sum_aggregation(predictions, k=20):
    preds = []
    for prediction in predictions:
        scores = None
        ans = []
        for i in range(k):
            res = prediction[i * 3: (i + 1) * 3]
            ans.append(answers[res.argmax()])
            if scores is not None:
                scores += res
            else:
                scores = res
            
        s = ans.count('SUPPORTS')
        r = ans.count('REFUTES')
        if s == 0 and r == 0:
            preds.append('NOT ENOUGH INFO')
        else:
            l = answers[scores[:2].argmax()]
            preds.append(l)
    return np.array(preds)


def catboost_aggregation(predictions):
    model = CatBoostClassifier()
    model.load_model('catboost_model20new.dump')
    
    test_pool = Pool(predictions)
    preds = model.predict(test_pool).flatten().astype(dtype=int).astype(dtype=object)
    preds[preds == 0] = 'SUPPORTS'
    preds[preds == 1] = 'REFUTES'
    preds[preds == 2] = 'NOT ENOUGH INFO'
    return preds


def get_sents(label, bert, pred):
    if label == 'NOT ENOUGH INFO':
        return []
    sents = set()
    #bert = bert.sort_values(by=['Relevance'], ascending=False)
    for i, r in bert.iterrows():
        res = np.array([r['SUPPORTS'], r['REFUTES'], r['NOT ENOUGH INFO']])
        title = pred.loc[i]['title']
        sent = pred.loc[i]['sent']
        text_sent = pred.loc[i]['text_a']
        if answers[np.argmax(res)] == label and len(sents) < 5:
            sents.add(tuple([title, sent, text_sent, answers[np.argmax(res)]]))
    for i, r in bert.iterrows():
        res = np.array([r['SUPPORTS'], r['REFUTES'], r['NOT ENOUGH INFO']])
        title = pred.loc[i]['title']
        sent = pred.loc[i]['sent']
        text_sent = pred.loc[i]['text_a']
        if len(sents) < 5:
            sents.add(tuple([title, sent, text_sent, answers[np.argmax(res)]]))
    return list(sents)


def get_sents_with_labels(preds, results, bert_res, pred):
    '''
    Args:
    preds - predicted labels for claims
    results - array of predicted sentences with scores for each claim (dict {title -> {num_sent -> score}})
    bert_res - bert prediction
    pred - data from the file passed for prediction in the model (contains fields text_a, text_b, sent, title, claim index)
    
    Returns:
    sentences with predictions coinciding with corresponding labels
    '''
    total = len(results)
    predict_sents = []
    predict_labels = []
    start = 0
    for i in tqdm(range(total)):
        end = start
        result = results[i]
        label = preds[i]
        if result == {}:
            predict_labels.append(answers[2])
            predict_sents.append([])
        else:
            while end < len(pred) and pred.iloc[end]['index'] == pred.iloc[start]['index']:    # search of the relevant part
                end += 1
            bert = bert_res.iloc[start: end - 1]
            start = end
            sents = get_sents(label, bert, pred)
            predict_labels.append(label)
            predict_sents.append(sents)
    return predict_labels, predict_sents


def fever_output_format(predict_labels, predict_sents, ids, out_path='results/predictions.jsonl'):
    '''
    Args:
    predict_labels = predicted labels
    predict_sents - predicted sentences
    ids - claims ids in test data set
    out_path - output path for json file
    
    Returns:
    json file in the FEVER-task output format
    '''
    if w_decoded == {}:
        load_wiki(wiki_path)
    
    total = len(predict_labels)
    with open(out_path, 'w', encoding='utf-8') as outfile:  
        for j in tqdm(range(total)):
            id = int(ids[j])
            predicted_label = predict_labels[j]
            pred_evidence = predict_sents[j]
            predicted_evidence = []
            
            if predicted_label == 'NOT ENOUGH INFO':
                title = None
                sent = None
            else:
                for pred_p in pred_evidence:
                    title = w_decoded[unidecode(pred_p[0])]
                    title = pred_p[0]
                    sent = int(pred_p[1])
                    predicted_evidence.append([title, sent])
            
            data = {
                    "id": id,
                    "predicted_label": predicted_label,
                    "predicted_evidence": predicted_evidence
                }
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

            
def aggregation(task_type, claims=None):
    '''
    Args:
    claims - array of claims for prediction
    task_type - "test" or "demo"
    
    Returns:
    list of predicted sentences and list of labels
    '''
    assert task_type in ['test', 'demo'], "the 'task_type' parameter must take one of two values: 'test' or 'demo'"
    assert task_type == 'test' or claims is not None, "you must provide claims list for 'demo' task type"
    
    bert_res = pd.read_csv('results/results_{}.csv'.format(task_type), header=None).rename(columns={0:"NOT ENOUGH INFO", 1:"SUPPORTS", 2:"REFUTES"})
    pred = pd.read_csv('results/pred_{}.tsv'.format(task_type), sep='\t')
    
    #rank = pd.read_csv('results/{}_ranking.csv'.format(task_type), header=None)[[1]].rename(columns={1:'Relevance'})
    
    with open('results/results_{}.pickle'.format(task_type), 'rb') as f:
        results = pickle.load(f)
    
    if task_type == 'test':
        train = pd.read_csv(data_path + "/FEVER_data/shared_task_test.csv") 
        claims = train.claim.values
        ids = train.id.values
    else:
        ids = np.arange(len(claims))
    
    predictions = get_prediction_format(claims, pred, bert_res)
    preds = catboost_aggregation(predictions)
    
    #bert_res = bert_res.join(rank)
    predict_labels, predict_sents = get_sents_with_labels(preds, results, bert_res, pred)
    fever_output_format(predict_labels, predict_sents, ids)


def train_and_eval_catboost(save_path='catboost_model.dump'):
    '''
    Args:
    save_path - path to save catboost model ("catboost_model.dump" as default)
    
    Returns:
    accuracy score and confusion matrix for eval set for the trained catboost model
    '''
    train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    train['evidence'] = train['evidence'].apply(lambda x: ast.literal_eval(x))
    claims = train.claim.values
    labels = train.label.values
    
    bert_res = pd.read_csv('results/results_eval.csv'.format(task_type), header=None).rename(columns={0:"NOT ENOUGH INFO", 1:"SUPPORTS", 2:"REFUTES"})
    pred = pd.read_csv('results/pred_eval.tsv'.format(task_type), sep='\t') # pred_{}TFIDFall
    
    predictions = get_prediction_format(claims, pred, bert_res)
    
    ground_truth = labels[:len(predictions)].copy()
    ground_truth[ground_truth == 'SUPPORTS'] = 0
    ground_truth[ground_truth == 'REFUTES'] = 1
    ground_truth[ground_truth == 'NOT ENOUGH INFO'] = 2
    
    #test_pred, predictions = predictions[:1000], predictions[1000:]   # one more validation set (the best model will be found on the traditional eval part)
    #test_gr, ground_truth = ground_truth[:1000], ground_truth[1000:]
    
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(predictions), pd.Series(ground_truth), test_size=0.3, random_state=57)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_test, y_test)
    
    model = CatBoostClassifier(iterations=100, task_type="GPU", depth=9, random_seed=42, eval_metric='Accuracy', loss_function='MultiClass')
    
    model.save_model(save_path)
 
    preds = catboost_aggregation(predictions)
    y_test[y_test == 0] = 'SUPPORTS'
    y_test[y_test == 1] = 'REFUTES'
    y_test[y_test == 2] = 'NOT ENOUGH INFO'
    
    return accuracy_score(y_test, preds), pd.DataFrame(confusion_matrix(y_test, preds, labels=answers), columns=answers, index=answers)


def calc_fever_score(claims, evidences, labels, results, bert_res, pred, algo, only_accuracy, save):
    assert algo in ['log', 'sum', 'boost'], "the 'algo' parameter must take one of three values: 'log', 'sum' or 'boost'"
    
    predictions = get_prediction_format(claims, pred, bert_res)
    
    if algo == 'boost':
        X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(predictions), pd.Series(np.zeros(len(predictions))), test_size=0.3, random_state=57)
        indixes = np.sort(np.array(X_test.index))
        preds = catboost_aggregation(predictions[indixes])
        labels = labels[indixes]
        results = results[indixes]
        cond = pred['index'].isin(indixes)
        pred = pred.loc[cond]
        bert_res = bert_res.loc[cond]
        claims = claims[indixes]
        evidences = evidences[indixes]
    elif algo == 'log':
        preds = logical_aggregation(predictions)
    else:
        preds = sum_aggregation(predictions)
    print('Accuracy:', accuracy_score(labels, preds))
    print(pd.DataFrame(confusion_matrix(labels, preds, labels=answers), columns=answers, index=answers))

    if not only_accuracy:
        predict_labels, predict_sents = get_sents_with_labels(preds, results, bert_res, pred)
        
        k = 0
        total = len(claims)
        for i in tqdm(range(total)):
            label = labels[i]
            predicted_label = predict_labels[i]
            evidence = evidences[i]
            predicted_evidence = predict_sents[i]
            if label == predicted_label:
                if label == 'NOT ENOUGH INFO':
                    k += 1
                else:
                    found = False
                    for p in evidence:
                        if len(p) == 1:    # evidence from several sentences is not considered
                            for pred_p in predicted_evidence:
                                if pred_p[1] == p[0][3] and unidecode(pred_p[0]) == unidecode(p[0][2]):
                                    found = True
                                    break
                    if found:
                        k += 1
        print('FEVER score:', k / total)
        
        if save:
            with open('results/evaluation_{}.pickle'.format(algo), 'wb') as f:
                pickle.dump(((predict_labels, predict_sents), (labels, evidences), claims), f)

    
def evaluation(algo='log', only_accuracy=False, save=False):
    '''
    Args:
    algo - algorithm for aggregation: "sum" (sum), "log" (logical) or "boost" (catboost)
    only_accuracy - in case of True only accuracy score will be calculated
    save - save answers from the evaluation step in the file results/evaluation_{algo}.pickle in the format ((predict_labels, predict_sents), (labels, evidences), claims)
    
    Returns:
    evaluation results - accuracy/FEVER scores
    '''
    assert algo in ['log', 'sum', 'boost'], "the 'algo' parameter must take one of three values: 'log', 'sum' or 'boost'"
    
    train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    train['evidence'] = train['evidence'].apply(lambda x: ast.literal_eval(x))
    
    claims = train.claim.values
    evidences = train.evidence.values
    labels = train.label.values
    
    with open('results/results_eval.pickle', 'rb') as f: # results_evalTFIDFall
        results = np.array(pickle.load(f))
        
    bert_res = pd.read_csv('results/results_eval.csv', header=None).rename(columns={0:"NOT ENOUGH INFO", 1:"SUPPORTS", 2:"REFUTES"}) # eval_resultsTFIDFall2
    pred = pd.read_csv('results/pred_eval.tsv', sep='\t') # pred_evalTFIDFall
    
    calc_fever_score(claims, evidences, labels, results, bert_res, pred, algo, only_accuracy, save)
