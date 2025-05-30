# Midprice Prediction for High-Frequency Trading

[![View Thesis (PDF)](https://img.shields.io/badge/View–Thesis-blue)](DLHFPP.pdf)

---

## 📖 Overview

This repository contains all the code and configuration used in my master’s thesis, **Deep Learning for High-Frequency Price Prediction in Cryptocurrency Markets**. The goal is to predict the log-return of the mid-price of 3 coins by leveraging CNN-LSTM networks on limit order book data.

## 📄 Thesis

The full write-up is available as a PDF:  
[DLHFPP.pdf](DLHFPP.pdf)

> *Deep Learning for High-Frequency Price Prediction.*  
> M.S. Thesis, University of Gothenburg, 2025.

---

## 🗂️ Repository Structure

```text
├── config/                     # YAML configs for data, experiments & models
│   ├── data_config.yaml        # raw data paths & loader settings
│   ├── experiment_config.yaml  # hyperparameters & training schedules
│   └── model_config.yaml       # network architecture settings
│
├── src/                        # Python source code
│   ├── data_utils.py           # data loading & preprocessing
│   ├── process_data.py         # per-symbol/month ETL orchestration
│   ├── feature_engineering.py  # LOB feature creation (CPU & GPU versions)
│   ├── model_architectures.py  # model definitions (CNN, LSTM, etc.)
│   ├── train_incremental.py    # incremental training loop
│   ├── evaluation.py           # metrics & quantitative evaluation
│   └── visualization.py        # plotting & result dashboards
│
├── DLHFPP.pdf                  # Master’s thesis document
└── .gitignore                  # ignore rules (e.g. data/, results/)
