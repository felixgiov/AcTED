# AcTED: Automatic Acquisition of Typical Event Duration for Semi-supervised Temporal Commonsense QA

## Data
We provide the generated pseudo-labeled QA data as specified in the paper under `data_generation/pseudo_qa_data/`.
Each file contains different number of events.

## Experiments

1. Generate the QA data for prediction with teacher model.
Please refer to `Section A` in `data_generation/README.md`

2. Fine-tune the model with McTACO-duration and predict the QA data from the previous step.
Please refer to `Section A` in `question_answering/README.md`

3. Generate the pseudo-labeled QA data.
Please refer to `Section B` in `data_generation/README.md`

4. Fine-tune the model with pseudo-labeled QA data,.
Please refer to `Section B` in `question_answering/README.md`

### Citation
If you use this dataset, please cite our paper ["Felix Virgo, Fei Cheng, Lis Kanashiro Pereira, Masayuki Asahara, Ichiro Kobayashi, Sadao Kurohashi. AcTED: Automatic Acquisition of Typical Event Duration for Semi-supervised Temporal Commonsense QA. arXiv preprint, (2024)"](https://arxiv.org/abs/2403.18504)

Bibtex for citations:

```bibtex
@misc{virgo2024acted,
      title={AcTED: Automatic Acquisition of Typical Event Duration for Semi-supervised Temporal Commonsense QA}, 
      author={Felix Virgo and Fei Cheng and Lis Kanashiro Pereira and Masayuki Asahara and Ichiro Kobayashi and Sadao Kurohashi},
      year={2024},
      eprint={2403.18504},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
