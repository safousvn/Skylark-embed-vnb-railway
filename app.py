import os
import time
import threading
import requests
import concurrent.futures

# ================= CONFIG =================
API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/embeddings/multimodal"
API_KEY = os.getenv("ARK_API_KEY")
MODEL = "skylark-embedding-vision-250615"

TARGET_TOKENS = 30_000_000
RUN_SECONDS = 5 * 3600  # 5 hours

REQUEST_INTERVAL = 3     # seconds
CONCURRENCY = 4
EST_TOKENS_PER_REQ = 4000
# ========================================

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Large text to inflate token count (vision + text)
LONG_TEXT = (
    "This is a detailed multimodal embedding load test. " * 800
)

payload = {
    "model": MODEL,
    "input": [
        {
            "type": "input_text",
            "text": LONG_TEXT
        },
        {
            "type": "input_image",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
        }
    ]
}

lock = threading.Lock()
total_tokens = 0
start_time = time.time()

# ========================================
def call_embedding(i):
    global total_tokens
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        data = r.json()

        usage = data.get("usage", {}).get("total_tokens", EST_TOKENS_PER_REQ)

        with lock:
            total_tokens += usage
            elapsed = time.time() - start_time
            tpm = int(total_tokens / elapsed * 60)

        print(
            f"[Req {i}] +{usage:,} tokens | "
            f"Total={total_tokens:,} | TPM≈{tpm:,}"
        )

    except Exception as e:
        print(f"[Req {i}] ❌ {e}")

# ========================================
def run_load():
    i = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        while True:
            elapsed = time.time() - start_time

            if elapsed >= RUN_SECONDS:
                print("⏱ 5 hours reached")
                break

            if total_tokens >= TARGET_TOKENS:
                print("🎯 Target tokens reached")
                break

            executor.submit(call_embedding, i)
            i += 1
            time.sleep(REQUEST_INTERVAL)

    print("\n✅ DONE")
    print(f"🔥 Total tokens: {total_tokens:,}")
    print(f"⏱ Runtime: {int(elapsed/60)} minutes")

# ========================================
if __name__ == "__main__":
    run_load()
