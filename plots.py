import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./solutions/problem1/cov_data/cov_data_pset3_e_0.6_0.5_300.csv")

# df = df[["fit_run_5", "fit_run_10", "fit_run_15", "fit_run_20", "fit_run_25"]]

# [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

data = df.iloc[list(range(5, 100, 5))]

print(data)

plt.boxplot(data.T)
plt.title("Average convergence rate for pset3_e_0.6_0.5_300")
plt.xlabel("Generations")
plt.ylabel("Fitness value")
plt.xticks(list(range(1, 20)), list(range(5, 100, 5)))

plt.savefig("./solutions/problem1/plots/cov_plot/pset3_e_0.6_0.5_300.png")


