# Data Generation
## A. Generate QA data for prediction with teacher model

1. Download and unzip one of the English Wikipedia dumps from https://dumps.wikimedia.org/enwiki/ and move it to `../tmp/wiki/` directory. 
    - In our paper, we use `enwiki-20210520-pages-articles.xml.bz2`.
2. Download the ConceptNet data from https://github.com/commonsense/conceptnet5/wiki/Downloads and move the file to `../tmp/` directory.
    - In our paper, we use `conceptnet-assertions-5.7.0.csv`.
3. To fetch event-sentences pairs, run:
```bash 
python fetch_sent_event_pairs.py --c ../tmp/conceptnet-assertions-5.7.0.csv --w ../tmp/wiki --o ../tmp
```
Alternatively, you can skip these steps and download the data here: https://lotus.kuee.kyoto-u.ac.jp/~felix/research/typical_duration/sent_event_pairs.zip. Then, unizp and move it to `../tmp/` directory.

4. To generate the QA data, run:
```bash 
python generate_qa_data_for_prediction.py --i ../tmp --o ../tmp
```
You can also skip this step and download the data here: https://lotus.kuee.kyoto-u.ac.jp/~felix/research/typical_duration/event_qa_to_predict.tsv and https://lotus.kuee.kyoto-u.ac.jp/~felix/research/typical_duration/event_sents_pairs.tsv. Then, move these files to `../tmp/` directory.

## B. Generate pseudo-labeled QA data
1. To generate the pseudo-QA data, run:
```bash 
python generate_pseudo_qa_data.py --e ../tmp/event_sents_pairs.tsv --p ../tmp/qa_prediction.txt --o ../tmp
```

The generated pseudo-QA from this section can be found here: `data_generation/pseudo_qa_data/`