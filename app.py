import requests, time, concurrent.futures, os

API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions"
API_KEY = os.getenv("ARK_API_KEY")  # store your API key as env var

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

prompt = "Write a long, detailed story about AI evolution, 5000 words minimum."

payload = {
    "model": "seed-1-6-flash-250715",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 4096,
}

def call_seed16(i):
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        data = response.json()
        usage = data.get("usage", {}).get("total_tokens", 0)
        print(f"Request #{i} consumed {usage} tokens.")
        return usage
    except Exception as e:
        print(f"Error in request #{i}: {e}")
        return 0

def run_load_test(concurrency=20, rounds=2000):
    total_tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(call_seed16, i) for i in range(rounds)]
        for f in concurrent.futures.as_completed(futures):
            total_tokens += f.result()
            print(f"Total tokens so far: {total_tokens:,}")
    print(f"✅ Load test complete. Total tokens: {total_tokens:,}")

if __name__ == "__main__":
    run_load_test(concurrency=20, rounds=2000)
