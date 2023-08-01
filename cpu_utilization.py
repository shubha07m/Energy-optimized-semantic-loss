import time
import csv
import psutil


def get_cpu_util():
    # Get the current CPU utilization
    utilization = psutil.cpu_percent()
    return [time.time(), utilization]


# Continuously print CPU utilization
with open('cpu_output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    while True:
        writer.writerow(get_cpu_util())
        time.sleep(1)
