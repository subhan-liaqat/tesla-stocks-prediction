# Tesla stock prediction (LSTM)

**Author:** [Subhan Liaqat](https://github.com/subhan-liaqat) · **Repository:** [subhan-liaqat/tesla-stocks-prediction](https://github.com/subhan-liaqat/tesla-stocks-prediction)

Personal ML project: I use **LSTM** recurrent networks on **Tesla (TSLA)** daily prices to study sequence modeling on financial time series. Outputs are **for learning only**, not trading or investment advice.

---

## Why it looks “from another era”

Nothing is wrong with your repo—three things read as **old on purpose**:

1. **The CSV stops in August 2017** (~nine years before 2026). It is a **fixed historical window** so experiments stay reproducible and lightweight. You are not “predicting today’s TSLA”; you are fitting a model on a closed slice.
2. **The notebooks use TensorFlow 1.x** (`tf.Session`, `tf.contrib.rnn`, …). That API is legacy, but it keeps the code aligned with classic graph-mode tutorials and makes the mechanics visible.
3. **Classic LSTM-on-stocks** was a hot demo topic around **2016–2018**, so the stack and dataset match that style—updated in **2026** in this repo with clearer docs and ownership.

---

## What’s in this repo

| File | Role |
|------|------|
| [`tesla_stocks.csv`](tesla_stocks.csv) | Daily OHLCV, **Aug 2014 → Aug 2017** (offline, no API keys). |
| [`tensorflow_lstm.ipynb`](tensorflow_lstm.ipynb) | **Practical path:** TF built-in LSTM cells, dropout, Adam—closest to a small “pipeline” style model. |
| [`lstm_from_scratch_tensorflow.ipynb`](lstm_from_scratch_tensorflow.ipynb) | **Learning path:** LSTM gates written out in TensorFlow for intuition. |

Theory refresher: [Understanding LSTMs (Colah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

---

## Environment

Use **TensorFlow 1.x** (e.g. 1.15) in a **Python 3.6–3.8**-style environment if you want to run cells without rewriting them. **TensorFlow 2** in eager mode will **not** run this code as-is.

Also: **Jupyter**, **NumPy**, **Pandas**, **Matplotlib**, **scikit-learn**.

---

## Run

From the project root (with `tesla_stocks.csv` next to the notebook):

```bash
jupyter notebook tensorflow_lstm.ipynb
```

```bash
jupyter notebook lstm_from_scratch_tensorflow.ipynb
```

---

## License

Released under the [MIT License](LICENSE).
