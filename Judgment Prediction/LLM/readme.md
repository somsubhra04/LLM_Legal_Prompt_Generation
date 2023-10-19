**JUDGEMENT PREDICTION- LLM (Brief writeup about the uploaded files)**

*Directory Path: Judgement Prediction > LLM > Datasets*


`JP.csv`

Contains the Judgement prediction only results of 256 test cases

---
`JP_with_pet_res.csv`

Similar to JP.csv, petitioner and respondent columns are also present.

---
`JPE.csv`

Contains the Judgement prediction and explanation results of 54 cases (56 expert annotated test cases out of which 2 cases
1951_10 and 1952_75 were used for training the system).

---
`JPE_with_pet_res.csv`

Similar to JPE.csv, petitioner and respondent columns are also present.

---
*Directory Path: Judgement Prediction > LLM > Codes*


`jp.py`

Contains the code that was used to find the predicted judgements by the LLM for 256 case proceedings.

---
`jpe.py`

Contains the code that was used to find judgement prediction for 54 case proceedings and their corresponding explanations.
