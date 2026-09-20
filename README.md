# Emotion Dataset Preprocessor

A small Streamlit app that builds a task dataset from the
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset on Hugging Face.
Sample tweets from each emotion, choose the ordering, and export the result with or without
ground-truth labels.

**Live app:** `https://emotion-subdataset-hai.streamlit.app/` <!-- replace after deploying -->

## Features

- Randomly sample a chosen number of tweets per emotion
- Reproducible results with a random seed
- Sequential (by emotion) or fully random ordering
- Ground-truth labels: none, in the main file, in a separate file, or both
- Label format: emotion name, numerical value, or both
- Export as CSV, JSON, or both

## How to use

1. Set **Tweets per emotion** and a **Random seed**.
2. Pick the **order** of the final dataset.
3. Choose where the **ground truth** goes and which **label format** to use.
4. Pick an **export format**, then click **Generate Task Dataset**.
5. Preview the result and download the files.

If participants should not see the answers, use **Separate file** and keep that file private.

## Output

| File              | Columns                                                     |
| ----------------- | ----------------------------------------------------------- |
| Main dataset      | `tweet_id`, `text` (+ ground-truth columns if included)     |
| Ground-truth file | `tweet_id`, `ground_truth_name` and/or `ground_truth_label` |

The numerical labels are the original dataset values:

| Value | Emotion  |
| ----- | -------- |
| 0     | sadness  |
| 1     | joy      |
| 2     | love     |
| 3     | anger    |
| 4     | fear     |
| 5     | surprise |

`tweet_id` is assigned after ordering, starting from 1.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py` and `requirements.txt` must be in the repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, then select the repository, branch, and main file (`app.py`).
4. Click **Deploy**.

Optional: to avoid Hugging Face rate-limit warnings, add a token under
**App settings → Secrets**:

```toml
HF_TOKEN = "your_huggingface_token"
```

## Project structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

## Credits

Dataset: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (Saravia et al., 2018).
