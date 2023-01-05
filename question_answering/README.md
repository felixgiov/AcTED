# Question Answering

This repository is tested on Python 3.8.3.
To install the required packages: 
```bash 
pip install -r requirements.txt
```

## A. Fine-tune the model with McTACO-duration and predict QA 

1. Download McTACO dataset here: https://github.com/CogComp/MCTACO and move the files to `../tmp/`.
2. To create a new dataset consisting of only the duration questions from McTACO and split the data for hold-out validation, run:
```bash
python extract_mctaco_duration.py --i ../tmp/ --o ../tmp/
```

3. To fine-tune RoBERTa-large model with the McTACO-duration data, run:
```bash
./finetune_mctaco.sh
```

Alternatively, you can download the fine-tuned model here; https://lotus.kuee.kyoto-u.ac.jp/~felix/research/commonsense_duration/roberta-large-mctaco-duration.zip. Then, unzip and move it to `../tmp/` directory.

4. To do QA prediction, run:
```bash
./predict.sh
```

## B. Fine-tune the model with pseudo-labeled QA data, then fine-tune it with McTACO-duration.
1. To fine-tune RoBERTa-large model with the pseudo-labeled QA data, run:
```bash
./finetune_pseudo.sh
```

2. To fine-tune the model from the previous step with the McTACO-duration data, run:
```bash
./finetune_mctaco_student.sh
```