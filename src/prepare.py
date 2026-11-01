from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    cfg = params["data"]
    input_path = ROOT / cfg["input"]
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    text_col = cfg["text_column"]
    score_col = cfg["score_column"]
    label_col = cfg["label_column"]
    identity_cols = list(cfg["identity_columns"])
    required = {text_col, score_col, *identity_cols}

    df = pd.read_csv(
        input_path,
        usecols=lambda col: col in required,
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[text_col] = df[text_col].fillna("").astype(str)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col]).copy()
    df[label_col] = (df[score_col] >= float(cfg["target_threshold"])).astype("int8")

    max_rows = cfg.get("max_rows")
    seed = int(cfg["random_state"])

    # Identity labels exist only on a subset of rows. Carve a fixed sample of
    # them out first so bias metrics are comparable across runs and the rows
    # never appear in train/valid, whatever max_rows is set to.
    has_identity = df[identity_cols].notna().all(axis=1)
    identity_pool = df[has_identity]
    eval_size = min(int(cfg["identity_eval_size"]), len(identity_pool))
    identity_eval = identity_pool.sample(n=eval_size, random_state=seed)
    df = df.drop(index=identity_eval.index)
    identity_eval = identity_eval[
        [text_col, score_col, label_col, *identity_cols]
    ].reset_index(drop=True)
    identity_eval.to_parquet(output_dir / "identity_eval.parquet", index=False)
    df = df[[text_col, score_col, label_col]]
    if max_rows is not None and len(df) > int(max_rows):
        df, _ = train_test_split(
            df,
            train_size=int(max_rows),
            random_state=seed,
            stratify=df[label_col],
        )

    df = df.reset_index(drop=True)
    train_df, valid_df = train_test_split(
        df,
        test_size=float(cfg["validation_size"]),
        random_state=seed,
        stratify=df[label_col],
    )

    # Keep the continuous toxicity score next to the binary label.
    train_df = train_df[[text_col, score_col, label_col]].reset_index(drop=True)
    valid_df = valid_df[[text_col, score_col, label_col]].reset_index(drop=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)

    print(f"Prepared {len(df):,} rows.")
    print(f"Training rows: {len(train_df):,}; validation rows: {len(valid_df):,}")
    print(f"Training toxic rate: {train_df[label_col].mean():.4f}")
    print(f"Validation toxic rate: {valid_df[label_col].mean():.4f}")
    print(
        f"Identity eval rows: {len(identity_eval):,} "
        f"(of {len(identity_pool):,} identity-labelled); "
        f"toxic rate: {identity_eval[label_col].mean():.4f}"
    )


if __name__ == "__main__":
    main()
