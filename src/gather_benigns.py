import hashlib
import os
import time
from pathlib import Path

import requests


API_URL = (
    "https://api.github.com/repos/"
    "iosifache/DikeDataset/contents/files/malware"
)

RAW_URL = (
    "https://raw.githubusercontent.com/"
    "iosifache/DikeDataset/main/files/malware"
)

OUTPUT_DIR = Path("data/external_test/malware")

TOTAL_SAMPLES = 820
CHUNK_SIZE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_exe_files():
    """Get all EXE files from the DikeDataset benign directory."""

    response = requests.get(
        API_URL,
        params={"per_page": 1000},
        timeout=60,
    )

    response.raise_for_status()

    files = response.json()

    exe_files = [
        item
        for item in files
        if item["type"] == "file"
        and item["name"].lower().endswith(".exe")
    ]

    return exe_files


def download_file(file_info):
    """Download one EXE and verify its SHA-256."""

    filename = file_info["name"]
    expected_hash = filename[:-4].lower()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    # Skip already downloaded files.
    if os.path.exists(output_path):
        print(f"    Already exists: {filename}")
        return True

    url = file_info["download_url"]

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=120,
        )

        response.raise_for_status()

        sha256 = hashlib.sha256()

        with open(output_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                f.write(chunk)
                sha256.update(chunk)

        actual_hash = sha256.hexdigest().lower()

        if actual_hash != expected_hash:

            print(
                f"    HASH MISMATCH!"
                f"\n    Expected: {expected_hash}"
                f"\n    Actual:   {actual_hash}"
            )

            os.remove(output_path)

            return False

        print(
            f"    OK: {filename}"
        )

        return True

    except Exception as e:

        print(
            f"    Download failed: {filename}"
            f"\n    {e}"
        )

        if os.path.exists(output_path):
            os.remove(output_path)

        return False


def main():

    print("Fetching DikeDataset benign file list...")

    files = get_exe_files()

    print(
        f"Found {len(files)} EXE files."
    )

    if len(files) < TOTAL_SAMPLES:
        raise RuntimeError(
            f"Only {len(files)} EXE files available."
        )

    # Take exactly 500.
    files = files[:TOTAL_SAMPLES]

    total_chunks = (
        TOTAL_SAMPLES + CHUNK_SIZE - 1
    ) // CHUNK_SIZE

    downloaded = 0

    for chunk_start in range(
        0,
        TOTAL_SAMPLES,
        CHUNK_SIZE,
    ):

        chunk = files[
            chunk_start:
            chunk_start + CHUNK_SIZE
        ]

        chunk_number = (
            chunk_start // CHUNK_SIZE
        ) + 1

        print()
        print("=" * 60)
        print(
            f"CHUNK {chunk_number}/{total_chunks}"
        )
        print("=" * 60)

        for index, file_info in enumerate(
            chunk,
            start=chunk_start + 1,
        ):

            print(
                f"[{index}/{TOTAL_SAMPLES}] "
                f"{file_info['name']}"
            )

            if download_file(file_info):
                downloaded += 1

            # Small delay between downloads.
            time.sleep(0.5)

        print(
            f"Chunk completed. "
            f"Downloaded: {downloaded}/{TOTAL_SAMPLES}"
        )

        # Delay between chunks.
        if chunk_start + CHUNK_SIZE < TOTAL_SAMPLES:
            print("Waiting 2 seconds...")
            time.sleep(2)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(
        f"Successfully downloaded: "
        f"{downloaded}/{TOTAL_SAMPLES}"
    )
    print(
        f"Directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()