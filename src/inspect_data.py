from pathlib import Path
import pandas as pd

RAW = Path("data/raw/DataCoSupplyChainDataset.csv")
OUT = Path("reports/stage1_data_report.txt")
OUT.parent.mkdir(exist_ok=True)

df = None
for enc in ("utf-8", "latin-1"):
    try:
        df = pd.read_csv(RAW, encoding=enc)
        print("Loaded with encoding:", enc)
        break
    except UnicodeDecodeError:
        continue
if df is None:
    raise SystemExit("Could not read the file. Check the path.")

with open(OUT, "w", encoding="utf-8") as f:
    def w(text=""):
        print(text)
        f.write(str(text) + "\n")

    w(f"Shape (rows, columns): {df.shape}")
    w("\nColumn summary:")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    w(info.to_string())
    w(f"\nExact duplicate rows: {df.duplicated().sum()}")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    w(f"\nDate-like columns: {date_cols}")
    for c in date_cols:
        d = pd.to_datetime(df[c], errors="coerce")
        w(f"{c}: {d.min()} -> {d.max()} (unparseable: {d.isna().sum()})")