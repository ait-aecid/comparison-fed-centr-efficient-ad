
import pandas as pd

results = pd.read_csv("results.csv")

print("Results")
print(results.head(6))

print("Gather results")
final = results.groupby(["Method", "N_Clients", "Dataset"]).mean()
print(final)

final.to_csv("final_results.csv")