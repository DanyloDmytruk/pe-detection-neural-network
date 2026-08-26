from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pefile
from tqdm import tqdm
import statistics


SAMPLES_DIR = Path("data/samples")
OUTPUT_FILE = Path("data/processed/features.parquet")


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def extract_section_features(pe):
    sections = pe.sections

    if not sections:
        return {
            "section_count": 0,
            "section_entropy_min": 0,
            "section_entropy_max": 0,
            "section_entropy_mean": 0,
            "section_entropy_std": 0,
            "executable_sections": 0,
            "writable_sections": 0,
            "readable_sections": 0,
            "rwx_sections": 0,
            "section_raw_size_min": 0,
            "section_raw_size_max": 0,
            "section_raw_size_mean": 0,
            "section_virtual_size_min": 0,
            "section_virtual_size_max": 0,
            "section_virtual_size_mean": 0,
            "section_size_ratio_min": 0,
            "section_size_ratio_max": 0,
            "section_size_ratio_mean": 0,
            "high_entropy_sections": 0,
        }

    entropies = []
    raw_sizes = []
    virtual_sizes = []
    size_ratios = []

    executable = 0
    writable = 0
    readable = 0
    rwx = 0
    high_entropy = 0

    for section in sections:
        entropy = section.get_entropy()

        raw_size = section.SizeOfRawData
        virtual_size = section.Misc_VirtualSize

        entropies.append(entropy)
        raw_sizes.append(raw_size)
        virtual_sizes.append(virtual_size)

        if raw_size > 0:
            ratio = virtual_size / raw_size
        else:
            ratio = 0

        size_ratios.append(ratio)

        characteristics = section.Characteristics

        is_executable = bool(characteristics & 0x20000000)
        is_readable = bool(characteristics & 0x40000000)
        is_writable = bool(characteristics & 0x80000000)

        if is_executable:
            executable += 1

        if is_readable:
            readable += 1

        if is_writable:
            writable += 1

        if is_executable and is_readable and is_writable:
            rwx += 1

        if entropy >= 7.0:
            high_entropy += 1

    return {
        # Existing features
        "section_count": len(sections),

        "section_entropy_min": min(entropies),
        "section_entropy_max": max(entropies),
        "section_entropy_mean": statistics.mean(entropies),
        "section_entropy_std": (
            statistics.stdev(entropies)
            if len(entropies) > 1
            else 0
        ),

        "executable_sections": executable,
        "writable_sections": writable,
        "readable_sections": readable,
        "rwx_sections": rwx,

        # New features
        "section_raw_size_min": min(raw_sizes),
        "section_raw_size_max": max(raw_sizes),
        "section_raw_size_mean": statistics.mean(raw_sizes),

        "section_virtual_size_min": min(virtual_sizes),
        "section_virtual_size_max": max(virtual_sizes),
        "section_virtual_size_mean": statistics.mean(virtual_sizes),

        "section_size_ratio_min": min(size_ratios),
        "section_size_ratio_max": max(size_ratios),
        "section_size_ratio_mean": statistics.mean(size_ratios),

        "high_entropy_sections": high_entropy,
    }


def extract_import_features(pe: pefile.PE) -> dict:
    dll_count = 0
    function_count = 0

    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        return {
            "imported_dlls": 0,
            "imported_functions": 0,
        }

    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll_count += 1
        function_count += len(entry.imports)

    return {
        "imported_dlls": dll_count,
        "imported_functions": function_count,
    }


def extract_pe_features(path: Path) -> dict:
    features = {
        "filename": path.name,
        "path": str(path),
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "valid_pe": False,
    }

    try:
        pe = pefile.PE(str(path), fast_load=False)

        features.update({
            "machine": pe.FILE_HEADER.Machine,
            "number_of_sections_header": pe.FILE_HEADER.NumberOfSections,
            "timestamp": pe.FILE_HEADER.TimeDateStamp,
            "characteristics": pe.FILE_HEADER.Characteristics,
            "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            "image_base": pe.OPTIONAL_HEADER.ImageBase,
            "subsystem": pe.OPTIONAL_HEADER.Subsystem,
        })

        features.update(extract_section_features(pe))
        features.update(extract_import_features(pe))

        features["valid_pe"] = True

        pe.close()

    except Exception as e:
        features["parse_error"] = str(e)

    return features


def collect_samples() -> list[tuple[Path, int]]:
    samples = []

    malware_dir = SAMPLES_DIR / "malware"
    benign_dir = SAMPLES_DIR / "benign"

    for path in malware_dir.rglob("*"):
        if path.is_file():
            samples.append((path, 1))

    for path in benign_dir.rglob("*"):
        if path.is_file():
            samples.append((path, 0))

    return samples


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()

    print(f"Found {len(samples)} samples")

    rows = []

    for path, label in tqdm(samples, desc="Extracting features"):
        features = extract_pe_features(path)
        features["label"] = label
        rows.append(features)

    df = pd.DataFrame(rows)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Samples: {len(df)}")
    print()
    print("Labels:")
    print(df["label"].value_counts())
    print()
    print("Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()