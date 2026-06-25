import pandas as pd

input_files = [
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_0.csv",
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_1000.csv",
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_2000.csv",
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_3000.csv",
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_4000.csv",
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_5000.csv",
    r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_batch_6000.csv",
]

output_file =  r"D:\SEGMain\SEGCode\Local-2-Global_SEG\index_output\rparis6k_self_pairs.csv"

dfs = []

for f in input_files:
    df = pd.read_csv(
        f,
        header=None,
        names=["source", "target", "score"]
    )
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

merged_df.to_csv(
    output_file,
    index=False,
    header=False
)

print(f"Saved: {output_file}")
print(f"Rows: {len(merged_df)}")
print(merged_df.head())