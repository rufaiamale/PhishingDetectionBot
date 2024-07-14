import pandas as pd

# Path to the dataset
dataset_path = 'dataset_full.csv'

# Load the dataset
df = pd.read_csv(dataset_path)

# Print the column names
print(df.columns)
