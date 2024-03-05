# VADER Sentiment Analysis

This repository contains a Python implementation of sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The code processes text reviews and determines whether the sentiment is positive or negative.


## Overview
This project performs sentiment analysis on a dataset of Amazon product reviews. The analysis uses the VADER sentiment analyzer, a rule-based tool that is especially effective for social media texts. The processed data is then used to classify the sentiment of each review as either positive or negative.

## Installation

To replicate this project, you'll need to have Python installed along with the necessary libraries. Follow the steps below to set up your environment:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Arch7399/sentiment-analysis.git
    cd sentiment-analysis
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```
    pandas
    nltk
    scikit-learn
    ```

4. **Download NLTK Data**:
    ```python
    import nltk
    nltk.download("all")
    ```

## Usage

To run the sentiment analysis, execute the Python script:

```bash
python sentiment_analysis.py
