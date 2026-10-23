"""Reproduce historical participant-level splits without changing ID ordering."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcpt_core import split_by_individual

SEEDS = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]

def main():
    rows = []
    summary = []
    for dataset, filename in [("Lab", "lab data.csv"), ("Online", "online data.csv")]:
        df = pd.read_csv(ROOT / "data" / filename)
        for seed in SEEDS:
            train, test, test_ids = split_by_individual(df, "ID", test_frac=0.2, seed=seed)
            assert set(train.ID).isdisjoint(set(test.ID))
            assert len(train) + len(test) == len(df)
            test_set = set(test_ids)
            for participant in df.ID.drop_duplicates():
                rows.append({"dataset":dataset,"seed":seed,"participant_id":participant,
                             "assignment":"test" if participant in test_set else "train"})
            summary.append({"dataset":dataset,"seed":seed,"train_individuals":train.ID.nunique(),
                            "test_individuals":test.ID.nunique(),"train_rows":len(train),"test_rows":len(test)})
    out = ROOT / "splits"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "split_manifest.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "split_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print("Historical split algorithm preserved; estimation scripts still compute their splits independently.")

if __name__ == "__main__":
    main()
