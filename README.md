# Tesla stock prediction with LSTMs

A small, notebook-first project where I experiment with **LSTM** networks on **Tesla (TSLA) historical prices**. The goal is to learn how recurrent models behave on real financial time series—not to publish trading signals. Treat every output as **educational**, not investment advice.

---

## What’s inside

| Piece | What it does |
|--------|----------------|
| [`tesla_stocks.csv`](tesla_stocks.csv) | Daily OHLCV history (**August 2014 → August 2017**), bundled so everything runs offline. |
| [`tensorflow_lstm.ipynb`](tensorflow_lstm.ipynb) | **Practical track:** TensorFlow’s built-in RNN stack (`BasicLSTMCell`, `dynamic_rnn`, dropout, Adam). This is the version you’d reach for when you want something closer to a standard pipeline. |
| [`lstm_from_scratch_tensorflow.ipynb`](lstm_from_scratch_tensorflow.ipynb) | **Learning track:** LSTM gates implemented explicitly—great for intuition before leaning on framework shortcuts. |

The model family is **RNN → LSTM** for sequence modeling of prices (scaled inputs, sliding windows, MSE-style training). If you want the intuition behind the cell diagram, [Christopher Olah’s LSTM post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is the classic reference.

---

## Environment

These notebooks were written against the **TensorFlow 1.x** API (`tf.contrib`, `tf.Session`, `tf.placeholder`, …). To run them as-is, use a Python environment with **TensorFlow 1.x** (for example TF 1.15 on a compatible Python). If you only have TensorFlow 2 installed, you’ll need either a separate env or a port of the graph/session code—**they won’t run unchanged on TF2 eager mode**.

**Typical stack**

- Python **3.6+** (3.6–3.8 often pairs well with older TF1 builds)
- [Jupyter](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/) with the Jupyter extension
- **NumPy**, **Pandas**, **Matplotlib**, **scikit-learn**, **TensorFlow 1.x**

Install the pieces you need, then open the repo folder so `tesla_stocks.csv` loads with a plain relative path.

---

## How to run

From the project root:

```bash
jupyter notebook tensorflow_lstm.ipynb
```

or, for the from-scratch LSTM:

```bash
jupyter notebook lstm_from_scratch_tensorflow.ipynb
```

Execute cells top to bottom. Training time depends on your CPU/GPU and batch settings.

---

## Why this repo exists (for me)

I keep this project around as a **clear, reproducible playground**: one dataset, two implementations, same underlying idea. It’s easy to come back months later and still remember where the “readable” LSTM ends and the “built-in cells” version begins.

---

## License

MIT License

Copyright (c) 2017 Luka Anicin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
