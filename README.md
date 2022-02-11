# MetaSieve



Implementation of _MetaSieve_ algorithm.
(For "MetaSieve: Quality vs. Complexity Sieve for Meta-Learningon Time Series Data" (KDD 2022))



## Content

- `total_prediction.py` - Generating synthetic data. Implementing brute-force calculations of quality metric for all generated sequences.
- `time_evaluation_LSTM.py` and `time_evaluation_RF.py` - files which contains code for comparing of evaluation time of brute-force method and _MetaSieve_.
- `meta_class_ser.ipynb` - file with the code for the classification algorithm which takes as input whole time-series.
- `meta_class_wvt.ipynb` - file with the code for the classification algorithm with wavelet transformations for time-series
- `metasieve_qual.ipynb` - Meta Sieve strategies illustrations.

## Data


- `artdata_350.csv` - generated 350 synthetic time-series.
- `real_data.csv` - real-world data consisting of stock value and electric consuption time-series.
- `res_artdata_RMSE.csv`, `res_artdata_sMAPE.csv`, `res_real_RMSE.csv`, `res_real_sMAPE.csv` - brute-forse sMAPE and RMSE results for sunthetic and real-world data.