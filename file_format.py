import pandas as pd
df = pd.read_csv("ai_job_dataset.csv")
df.to_parquet("ai_job_dataset.parquet",index=False)