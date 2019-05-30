# FEVER

This repository provides the solution for the Fact Extraction and VERification Shared Task.

## Requirement
* allennlp==0.8.3
* catboost
* numpy
* scipy
* scikit-learn
* tqdm
* mediawiki
* nltk
* tensorflow-gpu==1.13.1
* pandas
* pickle
* git+git://github.com/google-research/bert

## Work with the model

All instructions you can find in the Main.ipynb.

You should have the following folder structure:
```bash
data
├── FEVER_data
│   ├── train.csv
│   ├── shared_task_dev.csv
│   ├── shared_task_test.csv
│   └── wiki_pages.csv
└── FEVER_code
    ├── Main.ipynb
    ├── Convert_data.ipynb
    ├── Coreference.ipynb
    ├── dr.py
    ├── sr.py
    ├── nli.py
    ├── agg.py
    └── results
```
