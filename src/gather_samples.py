import os
import io
import requests
import pyzipper
from pathlib import Path
import time

API_URL = "https://mb-api.abuse.ch/api/v1/"
AUTH_KEY = "AUTH_KEY"

OUTPUT_DIR = Path("data/samples/malware")
TOTAL_SAMPLES = 493
CHUNK_SIZE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "Auth-Key": AUTH_KEY,
}


def api_request(data, retries=5):
    """Make a MalwareBazaar API request with retries."""

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=data,
                timeout=60,
            )

            if response.status_code == 200:
                return response

            print(
                f"    HTTP {response.status_code} "
                f"(attempt {attempt}/{retries})"
            )

        except requests.RequestException as e:
            print(
                f"    Request error: {e} "
                f"(attempt {attempt}/{retries})"
            )

        if attempt < retries:
            wait = attempt * 5
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    return None


def get_samples(limit=10):
    """Fetch EXE sample metadata."""

    response = api_request({
        "query": "get_file_type",
        "file_type": "exe",
        "limit": limit,
    })

    if response is None:
        return []

    try:
        result = response.json()
    except ValueError:
        print("    Invalid JSON response")
        return []

    if result.get("query_status") != "ok":
        print(f"    API error: {result}")
        return []

    return result.get("data", [])


def download_sample(sha256):
    """Download and extract one sample."""

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{sha256}.exe",
    )

    if os.path.exists(output_file):
        return "exists"

    response = api_request({
        "query": "get_file",
        "sha256_hash": sha256,
    })

    if response is None:
        return "failed"

    try:
        with pyzipper.AESZipFile(
            io.BytesIO(response.content)
        ) as archive:

            archive.pwd = b"infected"

            names = archive.namelist()

            if not names:
                return "empty"

            sample_data = archive.read(names[0])

            with open(output_file, "wb") as f:
                f.write(sample_data)

        return "downloaded"

    except Exception as e:
        print(f"    Extraction error: {e}")
        return "failed"


def main():

    downloaded = 0

    chunk_number = 0

    while downloaded < TOTAL_SAMPLES:

        chunk_number += 1

        remaining = TOTAL_SAMPLES - downloaded

        chunk_size = min(
            CHUNK_SIZE,
            remaining,
        )

        print()
        print("=" * 50)
        print(
            f"CHUNK {chunk_number} "
            f"| requesting {chunk_size} samples"
        )
        print("=" * 50)

        # Fetch metadata for this chunk.
        samples = get_samples(chunk_size)

        if not samples:
            print("No samples returned.")
            print("Waiting before retrying...")
            time.sleep(15)
            continue

        print(
            f"Received {len(samples)} samples"
        )

        # Download this chunk.
        for sample in samples:

            sha256 = sample["sha256_hash"]

            print(
                f"[{downloaded + 1}/{TOTAL_SAMPLES}] "
                f"{sha256}"
            )

            status = download_sample(sha256)

            print(f"    -> {status}")

            if status in ("downloaded"):
                downloaded += 1

            # Don't hammer the API.
            time.sleep(1)

        print(
            f"Progress: "
            f"{downloaded}/{TOTAL_SAMPLES}"
        )

        # Pause between chunks.
        if downloaded < TOTAL_SAMPLES:
            print("Waiting 5 seconds...")
            time.sleep(5)

    print()
    print("=" * 50)
    print("Finished!")
    print(f"Samples: {downloaded}")
    print(f"Directory: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()