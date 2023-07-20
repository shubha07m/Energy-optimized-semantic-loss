import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

plotdata = pd.read_csv('plotdata.csv')

Time = plotdata.loc[:, 'Time']

CPU_Power = plotdata.loc[:, 'CPU_Power']
GPU_Power = plotdata.loc[:, 'GPU_Power']
cpu_util = plotdata.loc[:, 'cpu_util']
System_instructions_retired = plotdata.loc[:, 'System_instructions_retired']
System_instructions_per_clock = plotdata.loc[:, 'System_instructions_per_clock']

# Smoothing : currently OFF
# smoothing Time_smooth = np.linspace(Time.min(), Time.max(), 700)  # Increase the number of data points for smoother
# interpolation CPU_Power_smooth = make_interp_spline(Time, CPU_Power)(Time_smooth) GPU_Power_smooth =
# make_interp_spline(Time, GPU_Power)(Time_smooth) cpu_util_smooth = make_interp_spline(Time, cpu_util)(Time_smooth)
# System_instructions_retired_smooth = make_interp_spline(Time, System_instructions_retired)(Time_smooth)
# System_instructions_per_clock_smooth = make_interp_spline(Time, System_instructions_per_clock)(Time_smooth)
#
# # Plotting the lines
# plt.plot(Time_smooth, CPU_Power_smooth, label='CPU_Power(dBm)')
# plt.plot(Time_smooth, GPU_Power_smooth, label='GPU_Power(dBm)')
# plt.plot(Time_smooth, cpu_util_smooth, label='cpu_util(%)')
# plt.plot(Time_smooth, System_instructions_retired_smooth, label='System_instructions_retired(log10)')
# plt.plot(Time_smooth, System_instructions_per_clock_smooth, label='System_instructions_per_clock')


# # Plotting the lines
plt.plot(Time, CPU_Power, label='CPU_Power(dBm)')
plt.plot(Time, GPU_Power, label='GPU_Power(dBm)')
plt.plot(Time, cpu_util, label='cpu_util(%)')
plt.plot(Time, System_instructions_retired, label='System_instructions_retired(log10)')
plt.plot(Time, System_instructions_per_clock, label='System_instructions_per_clock')

# Adding labels and title
plt.xlabel('Time(second)')
plt.ylabel('CPU and GPU resources')
plt.title('CPU and GPU utilization by Transformers')

vertical_lines = [27, 34, 39, 57, 62, 67, 72, 75, 79, 88]

for line in vertical_lines:
    plt.axvline(x=line, color='red', lw=.75, linestyle='--')


# Draw multiple horizontal lines  and texts - currently OFF
# plt.hlines(70, xmin=27, xmax=34, color='red', linestyle='-', linewidth=.5)
# plt.hlines(70, xmin=39, xmax=57, color='red', linestyle='-', linewidth=.5)
# plt.hlines(70, xmin=62, xmax=67, color='red', linestyle='-', linewidth=.5)
# plt.hlines(70, xmin=72, xmax=75, color='red', linestyle='-', linewidth=.5)
# plt.hlines(70, xmin=79, xmax=88, color='red', linestyle='-', linewidth=.5)

# plt.text(30, 70, 'vit-gpt2', fontsize=10, color='black', rotation='horizontal')
# plt.text(48, 70, 'git large', fontsize=10, color='black', rotation='horizontal')
# plt.text(65, 70, 'git base', fontsize=7, color='black', rotation='horizontal')
# plt.text(74, 70, 'blip base', fontsize=7, color='black', rotation='horizontal')
# plt.text(86, 70, 'blip large', fontsize=10, color='black', rotation='horizontal')


plt.text(30, 60, 'vit-gpt2', fontsize=10, color='red', rotation='vertical')
plt.text(48, 60, 'git large', fontsize=10, color='red', rotation='vertical')
plt.text(65, 45, 'git base', fontsize=7, color='red', rotation='vertical')
plt.text(74, 32, 'blip base', fontsize=7, color='red', rotation='vertical')
plt.text(86, 62, 'blip large', fontsize=10, color='red', rotation='vertical')


# Adding a legend
plt.legend(frameon=False)

# Displaying the graph
plt.show()
