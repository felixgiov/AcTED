# Semi-supervised Learning for Commonsense Duration QA

## Data
We provide the generated pseudo-labeled QA data as specified in the paper under `data_generation/pseudo_qa_data/`.
Each file contains different number of events.

## Experiments


1. Install the required packages: 
```bash 
pip install -r requirements.txt
```

2. Generate the QA data for prediction with teacher model.
Please refer to `Section A` in `data_generation/README.md`

3. Fine-tune the model with McTACO-duration and predict the QA data from the previous step.
Please refer to `Section A` in `question_answering/README.md`

4. Generate the pseudo-labeled QA data.
Please refer to `Section B` in `data_generation/README.md`

5. Fine-tune the model with pseudo-labeled QA data,.
Please refer to `Section B` in `question_answering/README.md`
