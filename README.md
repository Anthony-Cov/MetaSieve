# MetaSieve



Implementation of _MetaSieve_ algorithm.
(For "MetaSieve: Quality vs. Complexity Sieve for Meta-Learningon Time Series Data" (KDD 2022))



## Content

- `total_prediction.py` - Generating synthetic data. Implementing brute-force calculations of quality metric for all generated sequences.
- `time_evaluation_LSTM.py` and `time_evaluation_RF.py` - files which contains code for comparing of evaluation time of brute-force method and _MetaSieve_.
- `meta_class_ser.py` - file with the code for the classification algorithm which takes as input whole time-series.
- `meta_class_wvt.py` - file with the code for the classification algorithm with wavelet transformations for time-series



