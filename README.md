# Tesla stock prediction with LSTMs

**Subhan Liaqat** · [GitHub profile](https://github.com/subhan-liaqat)

LSTM models in TensorFlow on historical **Tesla (TSLA)** daily OHLCV data: sequence windows, scaled inputs, training loops, and prediction plots. The repo contains two notebooks—one using TensorFlow’s built-in LSTM stack (multi-layer, dropout, Adam), and one implementing LSTM gates explicitly in TensorFlow.

---

## Project structure

```
tesla-stocks-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── tesla_stocks.csv
├── tensorflow_lstm.ipynb
└── lstm_from_scratch_tensorflow.ipynb
```

| File | Description |
|------|-------------|
| `tesla_stocks.csv` | Daily OHLCV for TSLA (Aug 2014 → Aug 2017) |
| `tensorflow_lstm.ipynb` | LSTM with `tf.contrib.rnn`, stacked cells, dropout |
| `lstm_from_scratch_tensorflow.ipynb` | Same data; LSTM gates written out in TensorFlow |

---

## Setup

Use **Python 3.6–3.8** and a fresh virtual environment. Install dependencies:

```bash
pip install -r requirements.txt
```

The notebooks use **TensorFlow 1.x** (graph mode, `tf.Session`). They are written for that API and are not drop-in compatible with TensorFlow 2 eager execution without changes.

---

## Run

From the repository root:

```bash
jupyter notebook tensorflow_lstm.ipynb
```

```bash
jupyter notebook lstm_from_scratch_tensorflow.ipynb
```

Run cells top to bottom with `tesla_stocks.csv` in the same directory.

---

## Scope

Models are trained on the bundled historical file only (through 2017). This project is for **technical experimentation**, not trading or investment advice.

---

## License

[MIT License](LICENSE)
