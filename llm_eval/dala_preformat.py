"""Convert DaLA dataset to ;-separated CSV with id, Question, Answer."""
import os

import pandas as pd

dala_path = "../la_output/gen_v2/dala_da_gen_v2_val.csv"
df = pd.read_csv(dala_path)

# rename columns
df = df.rename(columns={"corrupted": "Question", "original": "Answer"})

# add id column
df.insert(0, "id", range(1, len(df) + 1))

# save in current folder adding _formatted to filename
output_path = os.path.splitext(dala_path)[0] + "_formatted.csv"
df.to_csv(output_path, index=False, sep=";")
print("Reformatted DaLA dataset saved to:", output_path)