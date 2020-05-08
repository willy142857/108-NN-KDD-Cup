# 108-NN-KDD-Cup

## KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing

### <https://tianchi.aliyun.com/competition/entrance/231785/introduction>

## Requirements

* python >= 3.7
* pandas >= 1.0.0
* tqdm == 4.45.0

## Evaluation

fake_test: generate by masking the latest ONE click made by each user in underexpose_test_click-T.csv

* fake_test/underexpose_test_T.csv
* fake_test/underexpose_test_qtime-T.csv
* fake_test/underexpose_test_qtime_with_answer-T.csv

usage:

```python
from evaluation import evaluate

# be sure you use fake_test to predict
evaluate(now_phase, submission_name)
```
