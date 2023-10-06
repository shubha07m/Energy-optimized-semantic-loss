import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plotdata = pd.read_csv('resource_plotdata_encoder.csv')

CPU_Power = plotdata.loc[:, 'CPU_Power']
GPU_Power = plotdata.loc[:, 'GPU_Power']

Time = [i for i in range(len(CPU_Power))]

# Normalize the data for CPU power and GPU power
# max_CPU_Power = max(CPU_Power)
# max_GPU_Power = max(GPU_Power)
# normalized_CPU_Power = [x / max_CPU_Power for x in CPU_Power]
# normalized_GPU_Power = [x / max_GPU_Power for x in GPU_Power]

# Create a new figure and axis for the second y-axis
fig, ax1 = plt.subplots()

# Plotting the normalized CPU power on the left y-axis
ax1.plot(Time, CPU_Power, label='CPU_Power', color='tab:orange')
ax1.set_xlabel('Time (second)', fontweight='bold', size=12)
ax1.set_ylabel('CPU power(W)', fontweight='bold', size=12, color='tab:orange')
ax1.tick_params(axis='y')

# Create a twin y-axis on the right side
ax2 = ax1.twinx()

# Plotting the GPU power on the right y-axis
ax2.plot(Time, GPU_Power, label='GPU_Power', color='tab:blue')
ax2.set_ylabel('GPU power(W)', fontweight='bold', size=12, color='tab:blue')

# Adding labels and title
plt.title('Power utilization during inference', fontweight='bold', size=12)

# Adding lines for the encoders

vertical_lines = [19, 27, 32, 70, 75, 90, 95, 101, 106, 114]
for line in vertical_lines:
    plt.axvline(x=line, color='red', lw=1, linestyle='--')
#
# # Adding legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines + lines2, labels + labels2, loc='best', frameon=False)

for text in legend.get_texts():
    text.set_fontweight('bold')

# Displaying the graph
plt.show()
