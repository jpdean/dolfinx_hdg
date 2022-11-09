import pandas as pd
import json
import matplotlib.pyplot as plt

# TODO Read from file instead
num_procs = [2**i * 76 for i in range(6)]

results = {}
for num_proc in num_procs:
    with open(f"results/stokes/results_{num_proc}.json", "r") as f:
        results_dict = json.load(f)

    for dict in results_dict.values():
        for key, val in dict.items():
            try:
                results[key].append(val)
            except KeyError:
                results[key] = [val]

results = pd.DataFrame(results)
# print(results)
# print(results.columns)

x = "num_proc"
results[[x, "its"]].plot.bar(x=x)
plt.tight_layout()
plt.show()

results[[x, "assemble_mat", "assemble_vec",
         "backsub", "assemble_pre"]].plot.bar(x=x, stacked=True)
plt.tight_layout()
plt.show()

results["other"] = results["total"] - results["solve"]
results[[x, "solve", "other"]].plot.bar(x=x, stacked=True)
plt.tight_layout()
plt.show()
