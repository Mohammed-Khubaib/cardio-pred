import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def dataprocessing(Main_Dataset):
    Chol_noise = Main_Dataset[Main_Dataset["chol"]>500].index
    for i in Chol_noise:
        Main_Dataset.drop(index=[i], inplace=True)
    Features = Main_Dataset.drop(columns='output')
    Features = pd.DataFrame(Features)
    scaler = MinMaxScaler()
    Norm_data = scaler.fit_transform(Features)
    Norm_df = pd.DataFrame(Norm_data, columns= Features.columns)

    return Main_Dataset , Norm_df