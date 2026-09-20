import pandas as pd
import streamlit as st
from datasets import load_dataset

DATASET_NAME = "dair-ai/emotion"

# order used when "Sequential by emotion" is picked
EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

st.set_page_config(
    page_title="Emotion Dataset Preprocessor",
    layout="wide",
)

st.title("Emotion Dataset Preprocessor")
st.write(
    "Build a task dataset from **dair-ai/emotion**: sample tweets per emotion, "
    "choose the order, and export with or without ground-truth labels."
)


# ---------- Load data ----------

# download once and reuse it on every rerun
@st.cache_data(show_spinner="Loading the emotion dataset...")
def load_emotion_data():
    train = load_dataset(DATASET_NAME)["train"]
    label_names = train.features["label"].names

    df = train.to_pandas()
    # add the emotion name next to the numeric label
    df["emotion"] = df["label"].map(dict(enumerate(label_names)))
    return df, label_names


try:
    tweets, label_names = load_emotion_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

# the smallest emotion sets how many tweets we can pick from each
max_per_emotion = int(tweets["emotion"].value_counts().min())

col1, col2 = st.columns(2)
col1.metric("Total tweets", f"{len(tweets):,}")
col2.metric("Emotions", len(label_names))

with st.expander("Numerical label mapping (original dataset values)"):
    st.dataframe(
        pd.DataFrame({"Numerical value": range(
            len(label_names)), "Emotion": label_names}),
        hide_index=True,
    )


# ---------- Settings ----------

st.subheader("1. Settings")

col1, col2 = st.columns(2)
per_emotion = col1.number_input(
    "Tweets per emotion",
    min_value=1,
    max_value=max_per_emotion,
    value=10,
    help=f"Tweets randomly picked from each emotion (max {max_per_emotion}).",
)
seed = col2.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42,
    help="The same seed gives the same selection and order.",
)

order = st.radio(
    "Order",
    ["Sequential by emotion", "Random order"],
    horizontal=True,
    help="Sequential: " + " → ".join(EMOTIONS),
)

st.subheader("2. Ground Truth")

gt_option = st.radio(
    "Where should ground-truth labels go?",
    ["None", "In main dataset", "Separate file", "Both"],
    horizontal=True,
)

# only ask about the label format if labels are being included
label_format = "Emotion name"
if gt_option != "None":
    label_format = st.radio(
        "Label format",
        ["Emotion name", "Numerical value", "Both"],
        horizontal=True,
    )

export_format = st.radio("3. Export format", [
                         "CSV", "JSON", "CSV + JSON"], horizontal=True)


# ---------- Helpers ----------

def build_dataset(df, per_emotion, seed, order):
    """Pick tweets for each emotion, set the order, and add ids."""
    parts = [
        df[df["emotion"] == emotion].sample(n=per_emotion, random_state=seed)
        for emotion in EMOTIONS
    ]
    # joining in EMOTIONS order already gives the sequential order
    result = pd.concat(parts, ignore_index=True)

    if order == "Random order":
        result = result.sample(
            frac=1, random_state=seed).reset_index(drop=True)

    # ids are added last so they follow the final order
    result.insert(0, "tweet_id", range(1, len(result) + 1))
    return result


def ground_truth_columns(df, label_format):
    """Build the label columns in the chosen format."""
    columns = pd.DataFrame(index=df.index)
    if label_format in ("Emotion name", "Both"):
        columns["ground_truth_name"] = df["emotion"].astype(str)
    if label_format in ("Numerical value", "Both"):
        columns["ground_truth_label"] = df["label"]
    return columns


def download_buttons(df, file_name, label):
    """Add CSV and/or JSON download buttons."""
    if export_format in ("CSV", "CSV + JSON"):
        st.download_button(
            f"Download {label} (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            f"{file_name}.csv",
            "text/csv",
            width="stretch",
        )
    if export_format in ("JSON", "CSV + JSON"):
        st.download_button(
            f"Download {label} (JSON)",
            df.to_json(orient="records", force_ascii=False,
                       indent=2).encode("utf-8"),
            f"{file_name}.json",
            "application/json",
            width="stretch",
        )


# ---------- Generate ----------

if st.button("Generate Task Dataset", type="primary", width="stretch"):
    dataset = build_dataset(tweets, per_emotion, seed, order)
    truth = ground_truth_columns(dataset, label_format)

    # main file is just the id and text, labels are optional
    main = dataset[["tweet_id", "text"]]
    if gt_option in ("In main dataset", "Both"):
        main = pd.concat([main, truth], axis=1)

    # separate file links labels back to tweets by id
    separate = None
    if gt_option in ("Separate file", "Both"):
        separate = pd.concat([dataset[["tweet_id"]], truth], axis=1)

    # keep the result so it stays visible after a download click
    st.session_state["result"] = {"main": main, "separate": separate}


# ---------- Results ----------

result = st.session_state.get("result")

if result:
    st.divider()
    st.subheader("Generated Dataset")
    st.caption(f"{len(result['main'])} tweets")

    st.dataframe(result["main"], height=400)
    download_buttons(result["main"], "emotion_task_dataset", "Main Dataset")

    if result["separate"] is not None:
        st.markdown("### Ground-Truth File")
        st.warning(
            "Keep this file private if participants should not see the labels.")
        st.dataframe(result["separate"], height=300)
        download_buttons(result["separate"],
                         "emotion_ground_truth", "Ground Truth")

st.divider()
st.caption("Dataset source: dair-ai/emotion on Hugging Face.")
