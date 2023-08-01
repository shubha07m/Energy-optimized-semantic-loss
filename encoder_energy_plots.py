import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

plotdata = pd.read_csv('plotdata.csv')

Time = plotdata.loc[:, 'Time']

CPU_Power = plotdata.loc[:, 'CPU_Power']
GPU_Power = plotdata.loc[:, 'GPU_Power']

# Smoothing : currently OFF
# smoothing Time_smooth = np.linspace(Time.min(), Time.max(), 700)  # Increase the number of data points for smoother
# interpolation CPU_Power_smooth = make_interp_spline(Time, CPU_Power)(Time_smooth) GPU_Power_smooth =
# make_interp_spline(Time, GPU_Power)(Time_smooth) cpu_util_smooth = make_interp_spline(Time, cpu_util)(Time_smooth)
# System_instructions_retired_smooth = make_interp_spline(Time, System_instructions_retired)(Time_smooth)
# System_instructions_per_clock_smooth = make_interp_spline(Time, System_instructions_per_clock)(Time_smooth)
#
# # Plotting the smoothed lines
# plt.plot(Time_smooth, CPU_Power_smooth, label='CPU_Power(dBm)')
# plt.plot(Time_smooth, GPU_Power_smooth, label='GPU_Power(dBm)')
# plt.plot(Time_smooth, cpu_util_smooth, label='cpu_util(%)')
# plt.plot(Time_smooth, System_instructions_retired_smooth, label='System_instructions_retired(log10)')
# plt.plot(Time_smooth, System_instructions_per_clock_smooth, label='System_instructions_per_clock')

# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)

# # Plotting the lines
plt.plot(Time, CPU_Power, label='CPU_Power(W)')
plt.plot(Time, GPU_Power, label='GPU_Power(W)')

# Adding labels and title
plt.xlabel('Time(second)', fontweight='bold', size=12)
plt.ylabel('CPU and GPU power (W)', fontweight='bold', size=12)
plt.title('Power utilization during inference', fontweight='bold', size=12)

vertical_lines = [27, 34, 39, 57, 62, 67, 72, 75, 79, 88]

for line in vertical_lines:
    plt.axvline(x=line, color='red', lw=1, linestyle='--')

plt.text(30, 20, 'VIT-gpt2', fontsize=12, color='red', rotation='vertical', fontweight='bold')
plt.text(48, 20, 'GIT large', fontsize=12, color='red', rotation='vertical', fontweight='bold')
plt.text(63, 15, 'GIT base', fontsize=12, color='red', rotation='vertical', fontweight='bold')
plt.text(72, 12, 'BLIP base', fontsize=12, color='red', rotation='vertical', fontweight='bold')
plt.text(82, 22, 'BLIP large', fontsize=12, color='red', rotation='vertical', fontweight='bold')

legend_font = {'weight': 'bold', 'size': 10}

# Adding a legend
plt.legend(loc='upper left', frameon=False, prop=legend_font)

# cpu_util = plotdata.loc[:, 'cpu_util']
# plt.subplot(2, 2, 2)
# plt.plot(Time, cpu_util, color='green',label='cpu_util(%)')
# # Adding labels and title
# plt.xlabel('Time(second)')
# plt.ylabel('CPU utilization (%)')
# plt.title('CPU utilization during inference')
# plt.legend(loc='upper left', frameon=False)
#
#
# System_instructions_retired = plotdata.loc[:, 'System_instructions_retired']
# plt.subplot(2, 2, 3)
# plt.plot(Time, System_instructions_retired, color='red', label='System_instructions_retired')
# plt.xlabel('Time(second)')
# plt.ylabel('System_instructions_retired')
# plt.title('System instructions retired during inference')
# plt.legend(loc='upper left', frameon=False)
#
#
# System_instructions_per_clock = plotdata.loc[:, 'System_instructions_per_clock']
# plt.subplot(2, 2, 4)
# plt.plot(Time, System_instructions_per_clock, color='cyan', label='System_instructions_per_clock')
# plt.xlabel('Time(second)')
# plt.ylabel('System_instructions_per_clock')
# plt.title('System instructions per clock during inference')
# plt.legend(loc='upper left', frameon=False)
#
#
# # Adjust layout to prevent overlapping titles and labels
# plt.tight_layout()

# Displaying the graph
plt.show()
