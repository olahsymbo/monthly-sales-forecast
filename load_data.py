import os
import pandas as pd


def load_data(root_path):
    data= []
    for entry in sorted(os.listdir(root_path)):
        if not entry.startswith('.') and os.path.isfile(os.path.join(root_path, entry)):
            datafile = os.path.join(root_path, entry)
            data_csv = pd.read_csv(datafile)
            data.append(data_csv)
    return data