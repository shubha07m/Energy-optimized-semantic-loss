import os
import datetime
import time
import csv
import subprocess as sp
import pandas as pd
import numpy as np
import math

# run this command in terminal, keeping any other applications than the intended one closed
# sudo powermetrics -i 1000 --samplers cpu_power,gpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info | grep -E -- 'Power|Sampled' >> powersample.txt
# Run the transformer resource comparison script to get timestamps
# Once testing completed, delete the summary section from the powersample.txt log
# TO-DO
# Then run the current script after following above steps only

data = pd.read_table('powersample.txt')

Time_ = []
CPU_Power = []
GPU_Power = []
Combined_Power = []
System_instructions_retired = []
System_instructions_per_clock = []

j = 0
for i in data.values:

    if 'Sampled' in i[0]:
        Time_.append(int(datetime.datetime.strptime(i[0][33:53], "%b %d %H:%M:%S %Y").timestamp()))

    if 'CPU Power' in i[0]:
        if float(i[0][11:-3]) == 0.0:
            CPU_Power.append(0.0)
        else:
            CPU_Power.append(10 * math.log10(float(i[0][11:-3])))

    if 'GPU Power' in i[0]:
        j += 1
        if j % 2:
            if float(i[0][11:-3]) == 0.0:
                GPU_Power.append(0.0)
            else:
                GPU_Power.append(10 * math.log10(float(i[0][11:-3])))

    if 'Combined Power' in i[0]:
        Combined_Power.append(10 * math.log10(float(i[0][34:-3])))

    if 'System instructions retired' in i[0]:
        System_instructions_retired.append(np.log10(float(i[0][29:])))

    if 'System instructions per clock' in i[0]:
        System_instructions_per_clock.append(i[0][31:])

df = pd.DataFrame({'Time': Time_, 'CPU_Power': CPU_Power, 'GPU_Power': GPU_Power,
                   'Combined Power': Combined_Power,
                   'System_instructions_retired': System_instructions_retired,
                   'System_instructions_per_clock': System_instructions_per_clock})

df.to_csv('powermetric.csv', index=False)
