import pandas as pd
import os

def read_generate_new_dataset(path):
    """seperate guidance and document from the dataset and generate new dataset
    """
    df = pd.read_csv(path)
    df['guidance_1'] = df['src'].apply(lambda x: x.split(':',maxsplit=1)[0])
    df['content'] = df['src'].apply(lambda x: x.split(':',maxsplit=1)[1])
    df.to_csv('/home/piyush.sar/Projects/LegalSifter/llm-edit/src/datasets/coedit_sampled_standardised.csv',index=False)
    
if __name__ == "__main__":
    read_generate_new_dataset('/home/piyush.sar/Projects/LegalSifter/llm-edit/src/datasets/coedit_sampled.csv')