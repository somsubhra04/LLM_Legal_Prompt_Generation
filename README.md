# LLMs – the Good, the Bad or the Indispensable?: A Use Case on Legal Statute Prediction and Legal Judgment Prediction on Indian Court Cases
This repository contains the relevant data and codes for the paper 'LLMs – the Good, the Bad or the Indispensable?: A Use Case on Legal Statute Prediction and Legal Judgment Prediction on Indian Court Cases' accepted to the Findings of the EMNLP 2023 conference.


```
LLM_Legal_Prompt_Generation
├── Judgment Prediction
│   ├── LLM
│   │   ├── Codes
│   │   │   ├── jp.py
│   │   │   ├── jpe.py
│   │   ├── Datasets
│   │   │   ├── JP.csv
│   │   │   ├── JPE.csv
│   │   │   ├── JPE_with_pet_res.csv
│   │   │   ├── JP_with_pet_res.csv
│   │   ├── readme.md
│   ├── Transformer based Models
│   │   ├── Codes
│   │   │   ├── Evalution on ILDC expert dataset.ipynb
│   │   │   ├── Legal_judgment_training_with_transformers.py
│   │   ├── Datasets
│   │   │   ├── readme.md
│   ├── surname_wordlist
│   │   ├── hindu_surname_file.txt
│   │   ├── muslim_surname_file.txt
├── Statute Prediction
│   ├── Baseline Models
│   │   ├── data_generator.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   ├── Model
│   │   │   ├── net.py
│   │   ├── Experiments
│   │   │   ├── params
│   │   │   │   ├── params_inlegalbert.json
│   │   │   │   ├── params_legalbert.json
│   │   │   │   ├── params_xlnet.json
│   ├── LLM
│   │   ├── Codes
│   │   │   ├── ALL TASK CODE.ipynb
│   │   │   ├── ALL TASK CODE.py
│   │   ├── Datasets
│   │   │   ├── 13_Cases_Gender and Bias Prediction_with explanations.csv
│   │   │   ├── 245cases.csv
│   │   │   ├── Gender and Religion Bias cases.csv
│   │   │   ├── query.csv
│   │   │   ├── statute_pred_100_cases_without_exp-gender_religion_bias.csv
│   │   │   ├── statute_pred_100_cases_without_exp.csv
│   │   │   ├── statute_pred_45_cases_with_exp.csv
│   │   │   ├── statute_pred_45_cases_without_exp.csv
│   │   ├── readme.md
├── README.md
```
