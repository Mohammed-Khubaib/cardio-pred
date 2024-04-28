import pandas as pd

def describe(Main_Dataset):
    description = Main_Dataset.describe()
    description.loc['dtypes'] = Main_Dataset.dtypes.astype(str)
    description.loc['unique'] = Main_Dataset.nunique()
    description.loc['missing values'] = Main_Dataset.isna().sum()
    description = description.transpose()
    description.reset_index(inplace=True)
    description.rename(columns={"index": "variable"}, inplace=True)
    description=description[['variable', 'dtypes', 'count','unique', 'mean', 'std', 'min', 'max']]
    return description
