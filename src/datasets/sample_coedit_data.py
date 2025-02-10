import kagglehub
import pandas as pd
import os

def get_dataset(sample_count):
    """Sample data from coedit dataset from kaggle.
    Coedit validation data has 3 tasktypes 'paraphrasing', 'gramatical correction', 'neutralize'.
    We sample few samples (sample_count) from each task type to create our dataset.
    Finally saving it as a .csv in current working directory.
    """
    path = kagglehub.dataset_download("thedevastator/coedit-nlp-editing-dataset")
    validation_data = pd.read_csv(f"{path}/validation.csv")
    sampled_df = validation_data.groupby("task").apply(lambda x: x.sample(n=sample_count, replace=False)).reset_index(drop=True)
    sampled_df.to_csv('coedit_sampled.csv')

if __name__ == "__main__":
    number_of_samples = 50
    get_dataset(number_of_samples)
