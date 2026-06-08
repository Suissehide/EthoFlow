import pandas as pd
from pathlib import Path

project = Path("D:\\LEO\\dlc-projects\\souris-bottomview-Leo-2026-06-05")
for h5 in project.glob("labeled-data/*/CollectedData_*.h5"):
    df = pd.read_hdf(h5)
    print(f"{h5.parent.name}")
    print(f"  Frames labellisées : {len(df)}")
    print(f"  Keypoints : {df.columns.get_level_values(1).unique().tolist()}")
    print(f"  Couverture par keypoint (% de frames où le point est placé) :")
    for kp in df.columns.get_level_values(1).unique():
        cov = df.xs(kp, axis=1, level=1).iloc[:, 0].notna().sum() / len(df) * 100
        print(f"    {kp:25s} : {cov:5.1f}%")