# PE Detection Neural Network

A machine-learning-based system for detecting potentially malicious Windows PE (Portable Executable) files.

The project investigates static PE analysis, feature engineering, API 2-gram analysis, external validation, and comparison with the EMBER2024 feature representation. It also provides a unified inference script capable of evaluating an executable with both the custom model and the EMBER2024-based model.

> **Note:** The current experimental classifiers are based on **LightGBM** (gradient boosting decision trees). The repository represents the machine-learning component of the broader master's thesis *“Development of a Neural Network Model for Detecting Potential Cyber Threats”*.

---

## Features

* Collection of malicious PE samples from MalwareBazaar
* Collection of benign PE samples from DikeDataset
* Static PE feature extraction
* PE section analysis
* Imported API analysis
* API 2-gram feature extraction
* LightGBM malware classification
* Confusion matrix and classification metrics
* Internal and external model validation
* EMBER2024 dataset preparation
* EMBER2024 feature vectorization
* Training of an EMBER2024-based LightGBM classifier
* Combined inference using two independent models
* Command-line `.exe` prediction

---

<img width="664" height="140" alt="image" src="https://github.com/user-attachments/assets/b1fb4ccf-a9d1-4e43-84e9-5133ac099b57" />

