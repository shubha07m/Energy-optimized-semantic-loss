import os
import datetime
import time
import csv
import subprocess as sp
import pandas as pd
import numpy as np
import math

# run this command in terminal, keeping any other applications than the intended one closed sudo powermetrics -i 1000
# --samplers cpu_power,gpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info | grep -E --
# 'Power|Sampled' >> powersample_encoder.txt Run the transformer resource comparison script to get timestamps Once
# testing completed, Then run the current script after following these steps only

data = pd.read_table('powersample_encoder.txt', header=None)

Time_ = []
CPU_Power = []
GPU_Power = []
Combined_Power = []
j = 0

for i in data.values:
    if 'Sampled system activity' in i[0]:
        Time_.append(int(datetime.datetime.strptime(i[0].split("(")[1].split(")")[0],
                                                    "%a %b %d %H:%M:%S %Y %z").timestamp()))

    if 'CPU Power' in i[0]:
        if i[0][11:-3] != '':
            CPU_Power.append(float(i[0][11:-3])/1000)

    if 'GPU Power' in i[0]:
        j += 1
        if j % 2:
            if i[0][11:-3] != '':
                GPU_Power.append(float(i[0][11:-3])/1000)

    if 'Combined Power' in i[0]:
        if i[0][34:-3] != '':
            Combined_Power.append(float(i[0][34:-3])/1000)

df = pd.DataFrame({'Time': Time_, 'CPU_Power': CPU_Power, 'GPU_Power': GPU_Power,
                   'Combined Power': Combined_Power})

# save as _decoder if using it for decoder side
df.to_csv('resource_plotdata_encoder.csv', index=False)
