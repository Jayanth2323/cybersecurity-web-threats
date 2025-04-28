import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'correlation_matrix' is your DataFrame containing the data
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
             annot=True,
             cmap="RdBu", 
             center=0,  # Centers the color scale at 0
             linewidths=0.5,
             linecolor='gray',
             cbar_kws={"shrink": .8})  # Shrink the color bar
plt.title("Correlation Matrix Heatmap")
plt.show()