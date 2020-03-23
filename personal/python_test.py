#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from time import sleep, time
max_cpu = multiprocessing.cpu_count()
print(f'{max_cpu} cpu\'s detected')


def poolworker(index):
    sleep(20)
    print(index)
    return index

n_jobs = 16
t0 = time()
pool = ProcessPoolExecutor(max_cpu)
futures = {} 
for i in range(n_jobs):
    futures[i] = pool.submit(poolworker, i)
results = {key:future.result() for key, future in futures.items()}

print(f'Time elapsed {int(time() - t0)}')
