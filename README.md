# Tesla (TSLA) price modeling with LSTMs

**Subhan Liaqat** · [GitHub](https://github.com/subhan-liaqat) · [Repository](https://github.com/subhan-liaqat/tesla-stocks-prediction)

End-to-end **deep learning** exercise: I built **long short-term memory (LSTM)** models in **TensorFlow** to predict next-step prices from sequences of historical **Tesla** daily OHLCV data. The focus is **methodology**—preprocessing, windowing, training loops, and comparing a library LSTM stack to a **from-scratch** gate implementation—not production trading.

---

## At a glance

| | |
|---|---|
| **Domain** | Time-series forecasting, recurrent neural networks |
| **Data** | Daily OHLCV, Aug 2014 → Aug 2017 (bundled CSV, no external APIs) |
| **Models** | Stacked LSTM with dropout; optional manual LSTM cell (gates in TensorFlow) |
| **Stack** | Python · TensorFlow 1.x (graph / `Session`) · NumPy · Pandas · scikit-learn · Matplotlib · Jupyter |
| **Deliverables** | Two runnable notebooks + dataset + license |

---

## Why this project (what it shows)

- **Sequence modeling:** Sliding windows over scaled price series; supervised next-step targets.
- **Two implementations:** (1) **High-level** `tf.contrib.rnn` LSTM cells, multi-layer RNN, dropout, Adam, gradient clipping. (2) **Low-level** LSTM gates expressed in TensorFlow for understanding.
- **Full training loop:** Epochs, batching, loss minimization (MSE-style objectives in graph), monitoring loss over training (example runs are saved in the notebooks).
- **Reproducibility:** Fixed historical slice and offline data so results are comparable across runs and machines.

This is appropriate to discuss in interviews as a **learning and experimentation** project. It is **not** a trading system, and past performance on old data does not imply future results.

---

## Repository layout

| File | Description |
|------|-------------|
| [`tesla_stocks.csv`](tesla_stocks.csv) | Daily open, high, low, close, volume for TSLA |
| [`tensorflow_lstm.ipynb`](tensorflow_lstm.ipynb) | Production-style path: built-in LSTM, stacked layers, dropout |
| [`lstm_from_scratch_tensorflow.ipynb`](lstm_from_scratch_tensorflow.ipynb) | Educational path: LSTM internals implemented explicitly |

**Further reading:** [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Christopher Olah).

---

## Skills you can map to a resume

- Time-series preprocessing and **feature scaling** (e.g., `StandardScaler`)
- **Recurrent neural networks** and **LSTM** architecture
- **TensorFlow** graph construction, sessions, placeholders, optimization
- **Experiment workflow** in **Jupyter** notebooks
- Clear documentation of **limitations** and **ethical scope** (non-advisory use)

---

## Setup and run

**Requirement:** These notebooks target **TensorFlow 1.x** (e.g. 1.15) with **Python 3.6–3.8** in a dedicated environment. They do **not** run unchanged on TensorFlow 2 eager execution without modification.

```bash
# From the repository root (ensure tesla_stocks.csv is in the same folder)
jupyter notebook tensorflow_lstm.ipynb
jupyter notebook lstm_from_scratch_tensorflow.ipynb
```

Install TensorFlow 1.x, Jupyter, NumPy, Pandas, Matplotlib, and scikit-learn in that environment, then execute cells in order.

---

## Limitations (honest scope)

- **Data window:** The CSV ends in **2017**; models are fit on that era only—useful for methods, not for forecasting today’s market.
- **Framework:** Code uses the **TensorFlow 1** API by design for explicit graphs; migrating to **TF2 / Keras** is a natural follow-up project.
- **Purpose:** Research and learning. Not financial advice.

---

## License

Released under the [MIT License](LICENSE).
