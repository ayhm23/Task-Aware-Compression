import os, json, pandas as pd

METRICS_DIR = "results/metrics"

def load_existing():
    return pd.read_csv(os.path.join(METRICS_DIR, "full_results_table.csv"))

def append_pca(df):
    rows = []
    for dim in [32, 64, 128, 256]:
        path = os.path.join(METRICS_DIR, f"pca_dim{dim}.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        rows.append({
            "Method": "pca", "Dim": dim, "Mode": "pca", "TrainTask": "agnostic",
            "STS_spearman": r.get("sts", {}).get("spearman"),
            "NLI_accuracy": r.get("nli", {}).get("accuracy"),
            "CLS_accuracy": r.get("classification", {}).get("accuracy"),
        })
    if len(rows) == 0:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

def append_mixed(df):
    rows = []
    for pair in [("sts", "nli"), ("nli", "classification")]:
        for dim in [32, 64, 128, 256]:
            ta, tb = pair
            path = os.path.join(METRICS_DIR, f"autoencoder_mixed_{ta}_{tb}_dim{dim}.json")
            if not os.path.exists(path):
                continue
            r = json.load(open(path))
            rows.append({
                "Method": "autoencoder", "Dim": dim,
                "Mode": "mixed", "TrainTask": f"{ta}+{tb}",
                "STS_spearman": r.get("sts", {}).get("spearman"),
                "NLI_accuracy": r.get("nli", {}).get("accuracy"),
                "CLS_accuracy": r.get("classification", {}).get("accuracy"),
            })
    if len(rows) == 0:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

df = load_existing()
df = append_pca(df)
df = append_mixed(df)
df.drop_duplicates(subset=["Method", "Dim", "Mode", "TrainTask"], keep="last", inplace=True)
df.to_csv(os.path.join(METRICS_DIR, "full_results_table.csv"), index=False)
print(f"Updated table: {len(df)} rows")
print(df[df["Mode"].isin(["pca","mixed"])].to_string(index=False))
