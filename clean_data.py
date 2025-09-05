import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- CONFIG ---
INPUT_FOLDER = "data"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "cleaned")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Optional macro data used for additional features
MACRO_PATH = os.path.join(INPUT_FOLDER, "macro.csv")
if os.path.exists(MACRO_PATH):
    macro_df = pd.read_csv(MACRO_PATH, parse_dates=["QUOTE_DATE"])
else:
    macro_df = None

def clean_and_export_csv(file_path):
    filename = os.path.basename(file_path)
    stem     = os.path.splitext(filename)[0]

    print(f"→ Loading {filename}")
    df = pd.read_csv(file_path)

    # 0) merge macro features if available
    if macro_df is not None and "QUOTE_DATE" in df.columns:
        df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
        df = df.merge(macro_df, on="QUOTE_DATE", how="left")

    # 1) Recompute moneyness correctly for calls vs. puts
    if {"UNDERLYING_LAST","STRIKE","is_call"}.issubset(df.columns):
        df["moneyness"] = np.where(
            df["is_call"] == 1,
            df["UNDERLYING_LAST"] / df["STRIKE"],    # call: S/K
            df["STRIKE"] / df["UNDERLYING_LAST"]     # put:  K/S
        )

    # 2) Drop rows with any missing values
    df = df.dropna()

    # 3) Core filters
    if "VOLUME" in df.columns and "price" in df.columns:
        df = df[(df["VOLUME"] > 0) & (df["price"] > 0)]

    if "ttm" in df.columns:
        df = df[df["ttm"] > 0]

    if "IV" in df.columns:
        df = df[(df["IV"] > 0) & (df["IV"] <= 10)]

    if "DELTA" in df.columns:
        df = df[df["DELTA"].abs() <= 1]

    if "GAMMA" in df.columns:
        df = df[df["GAMMA"] >= 0]

    if "VEGA" in df.columns:
        df = df[df["VEGA"] >= 0]

    if "moneyness" in df.columns:
        df = df[(df["moneyness"] > 0) & (df["moneyness"] <= 3)]

    if df.empty:
        print(f"⚠ Skipped {filename} — no data left after cleaning")
        return

    # 4) Save cleaned CSV
    out_csv = os.path.join(OUTPUT_FOLDER, f"{stem}.csv")
    df.to_csv(out_csv, index=False)

    # 5) Save cleaned Parquet
    out_parquet = os.path.join(OUTPUT_FOLDER, f"{stem}.parquet")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_parquet, compression="snappy")

    print(f"✓ Saved cleaned: {out_csv} and {out_parquet}")

def main():
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "spx*.csv"))
    for file_path in all_files:
        clean_and_export_csv(file_path)
    print("\n✅ All files cleaned and saved to:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
