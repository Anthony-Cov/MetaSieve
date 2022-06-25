# MetaSieve



Implementation of _MetaSieve_ algorithm.
(For "MetaSieve: Quality vs. Complexity Sieve for Meta-Learningon Time Series Data" (KDD 2022))



## predictions

- `LSTM_prediction.py` - Generating synthetic data. Implementing brute-force calculations of quality metric for all generated sequences.
- `LSTM_prediction.py`, `RF_prediction.py`, and `XGB_prediction.py` - files which contains code for obtaining predictions accuracy for 15 levels of LSTM, RF, and XGBoost models respectivly.


## results
- `acc_time.ipynb` - Notebook, which provides the research  of time efficiency of different strategies of  _MetaSieve_.
- `seive_drawing.ipynb` - Notebook, which provides the research  of  _MetaSieve_ results with the usage of different _quality control strategies_.
- `GNNclass.ipynb` - realization of GNN classifier for the Second sttage of _MetaSieve_.

## storage

- `artdata_1000.csv` - generated 1000 synthetic time-series.
- `real_data.csv` - real-world data consisting of stock value and electric consuption time-series.
- `art1000_LSTM_acc_time.csv`, `art1000_RF_acc_time.csv`, `art1000_XGB_acc_time.csv` - brute-forse sMAPE and RMSE results for synthetic data with measured time.

## other 
- `ArtComposer.py` - generation process of 1000 synthetic time-series.

