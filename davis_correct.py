import pandas as pd
import numpy as np
# Load the data
data = pd.read_csv('data/davis/davis.txt', sep=',')
data["Y"]  = data["Y"].apply(lambda x: -np.log10(x/1e9))
data.to_csv('data/davis/davis.txt', index=False)