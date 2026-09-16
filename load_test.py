import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

URL = "https://ml-churn-end-to-end.onrender.com/predict"

customer = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000
}

TOTAL_REQUESTS = 100
CONCURRENT_USERS = 10


def send_request():
    start = time.perf_counter()

    try:
        response = requests.post(URL, json=customer, timeout=30)
        latency = time.perf_counter() - start

        return response.status_code, latency

    except Exception:
        latency = time.perf_counter() - start
        return None, latency


start_time = time.perf_counter()

results = []

with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:

    futures = [
        executor.submit(send_request)
        for _ in range(TOTAL_REQUESTS)
    ]

    for future in as_completed(futures):
        results.append(future.result())


total_time = time.perf_counter() - start_time

successful = [
    latency for status, latency in results
    if status == 200
]

failed = [
    latency for status, latency in results
    if status != 200
]


print("\n========== LOAD TEST RESULTS ==========")

print(f"Total requests:       {TOTAL_REQUESTS}")
print(f"Successful requests:  {len(successful)}")
print(f"Failed requests:      {len(failed)}")

print(f"Error rate:            {len(failed) / TOTAL_REQUESTS * 100:.2f}%")

if successful:

    print(f"Average latency:       {statistics.mean(successful):.4f}s")
    print(f"p50 latency:           {statistics.median(successful):.4f}s")

    sorted_latency = sorted(successful)

    p95_index = int(0.95 * len(sorted_latency)) - 1
    p99_index = int(0.99 * len(sorted_latency)) - 1

    print(f"p95 latency:           {sorted_latency[p95_index]:.4f}s")
    print(f"p99 latency:           {sorted_latency[p99_index]:.4f}s")

print(f"Total test time:       {total_time:.4f}s")
print("========================================")