import wfdb
import pandas as pd

# Path to ECG record (without extension)
record_path = "./ecg_data/100"

# Load raw ECG signal
record = wfdb.rdrecord(record_path, channel_names=['MLII'])
signal = record.p_signal[:,0]

# Load annotations
annotation = wfdb.rdann(record_path, 'atr')

# Build DataFrame with annotations aligned to sample index
df = pd.DataFrame({
    "Sample": range(len(signal)),
    "Amplitude": signal,
    "Annotation": ""
})
for samp, sym in zip(annotation.sample, annotation.symbol):
    if samp < len(df):
        df.at[samp, "Annotation"] = sym

# Show a few rows around each annotation for context
rows_list = []
for samp, sym in zip(annotation.sample, annotation.symbol):
    start = max(0, samp-3)
    end = min(len(df), samp+4)
    snippet = df.iloc[start:end].copy()
    snippet["Context"] = f"Beat {sym} at {samp}"
    rows_list.append(snippet)

df_context = pd.concat(rows_list)

# Summarize annotations
summary = pd.Series(annotation.symbol).value_counts().reset_index()
summary.columns = ["Annotation", "Count"]

# Display results
print("=== Annotation Summary ===")
print(summary)
print("\n=== Contextual Rows Around Each Annotation ===")
print(df_context.head(500))  # print first 50 rows for brevity
