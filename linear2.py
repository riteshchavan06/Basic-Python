import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
X = np.arange(1, 51)  # Values from 1 to 50
Y = 2.5 * X + np.random.normal(0, 5, size=len(X))  # Linear relation with noise

# Create DataFrame
df = pd.DataFrame({'X': X, 'Y': Y})

# Save to CSV
df.to_csv("data1.csv", index=False)

# Visualize with Seaborn
sns.set_style("darkgrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["X"], y=df["Y"], color="blue", label="Random Data")
sns.regplot(x="X", y="Y", data=df, scatter=False, color="red", label="Regression Line")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.title("Random Data with Regression Line")
plt.show()

print("Random data saved as data1.csv!")
