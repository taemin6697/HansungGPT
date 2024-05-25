from datasets import Dataset
import pandas as pd

# Load the CSV file
file_path = 'generated_data/qa_pairs_split_individual.csv'
df = pd.read_csv(file_path)

# Convert the DataFrame to a Huggingface Dataset
dataset = Dataset.from_pandas(df)

# Push the dataset to the Huggingface Hub
dataset.push_to_hub("kfkas/hansung_data_v2", private=False)
