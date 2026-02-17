import os, time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

path = "/lustre/alice/users/csonnab/PhD/jobs/clusterization/QA/output/24012026_PbPb_aMC_24arp2_559843_newTracking_clusterError/gpu_cf/reco/dump_cluster_error.csv"

s1 = time.time()
df = pd.read_csv(path)  # add sep=',' if needed
df = df.iloc[:-1]
df = df.astype(np.float32)
s2 = time.time()
print(f"Time to read data: {s2 - s1:.2f} seconds")

counts = df["cluster.num"].value_counts().sort_index()

print("Maximum cluster number counts:", np.max(counts))

hist_counts, bin_edges = np.histogram(counts, bins=np.arange(1, np.max(counts) + 2) - 0.5)
fig = plt.figure(figsize=(8, 6))
plt.bar(bin_edges[:-1], hist_counts, width=1)
plt.xlabel("cluster.num occurrence")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.dirname(path) + "/cluster_num_distribution.pdf")
plt.close()

print("Peak cluster number counts:", np.argmax(hist_counts))

df_filtered = df[df["cluster.num"].map(df["cluster.num"].value_counts()) >= np.argmax(hist_counts)]

latest_clusters = df_filtered.groupby("cluster.num", sort=False).tail(1)

mask_valid = (np.abs(latest_clusters["mP[0]"]) < 400) & (np.abs(latest_clusters["mP[1]"]) < 400)
latest_clusters = latest_clusters[mask_valid]

latest_clusters.to_csv(os.path.dirname(path) + "/latest_clusters.csv", index=False)

# data_Y = np.zeros((2, len(latest_clusters)), dtype=np.float32)
# data_Y[0] = np.abs(latest_clusters["clusterY"].to_numpy() - latest_clusters["mP[0]"].to_numpy())**2 - latest_clusters["mC[0]"].to_numpy()**2
# data_Y[1] = np.abs(latest_clusters["clusterZ"].to_numpy() - latest_clusters["mP[1]"].to_numpy())**2 - latest_clusters["mC[2]"].to_numpy()**2
#
# x_labels = "clusterState,clusterY,clusterZ,mP[0],mP[1],mP[2],mP[3],mP[4],mC[0],mC[2],mC[5],mC[9],mC[14]".split(",")
# data_X = np.zeros((5, len(latest_clusters)), dtype=np.float32)