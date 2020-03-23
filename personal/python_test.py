#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from time import sleep, time
max_cpu = multiprocessing.cpu_count()
use_cpu = min(16, max_cpu)
print(f'{max_cpu} cpu\'s available, using {use_cpu}')


def poolworker(job_executed):
    print(job_executed)
    sleep(10)
    return job_executed

n_jobs = 15


t0 = time()
pool = ProcessPoolExecutor(max_workers=use_cpu)
futures = {} 
for i in range(n_jobs):
    futures[i] = pool.submit(poolworker, i+1)
results = {key:future.result() for key, future in futures.items()}
print(f'Time elapsed {int(time() - t0)}')
