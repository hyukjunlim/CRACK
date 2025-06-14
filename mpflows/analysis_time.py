import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

# Set filename based on mode
filename = 'save_logs/4/s2ef_predictions.npz'

# Load and reshape data
data = np.load(filename)
print(data.files)


# print(data["time_first"].shape)
# print(data["time_last"].shape)

# print(f'Time first: {data["time_first"].mean():.4f} ± {data["time_first"].std():.4f} s')
# print(f'Time last: {data["time_last"].mean():.4f} ± {data["time_last"].std():.4f} s')

# print(f'Ratio: {data["time_last"].mean() / data["time_first"].mean():.4f}')

# # save the printed results to a file
# with open('flow_output/time_analysis.txt', 'w') as f:
#     f.write(f'Time first: {data["time_first"].mean():.4f} ± {data["time_first"].std():.4f} s\n')
#     f.write(f'Time last: {data["time_last"].mean():.4f} ± {data["time_last"].std():.4f} s\n')
#     f.write(f'Ratio: {data["time_last"].mean() / data["time_first"].mean():.4f}\n')
