# Email Content Extraction and Summarization

## Overview
This project focuses on developing an AI-driven framework for **email content extraction and summarization**, leveraging both **unsupervised** and **supervised learning techniques**. Conducted in collaboration with **RADD**, it addresses challenges in processing unstructured email data by employing advanced deep learning models and innovative preprocessing methods.

## Features
- **Preprocessing Pipeline**: Removes noise, standardizes dates, times, and email IDs, and handles HTML formatting.
- **Clustering**: Uses TF-IDF and BERT embeddings to group emails into meaningful clusters.
- **Redundancy Detection**: Eliminates exact and partial duplicates using MD5 and TF-IDF hashing.
- **Extractive Summarization**: Implements algorithms like TextRank and ShortMail for key information extraction.
- **Abstractive Summarization**: Fine-tunes transformer models like T5, BART, and LongT5 to generate coherent summaries.
- **Hybrid Approach**: Combines extractive and abstractive techniques for improved results.
- **Evaluation Metrics**: Utilizes ROUGE and BERT scores for performance assessment.
- **GDPR Compliance**: Ensures data security and privacy standards.
- **Visualization**: Interactive dashboards for data trends and insights.

## Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - TensorFlow, PyTorch, Hugging Face Transformers
  - BeautifulSoup, NLTK, Scikit-learn
- **Pretrained Models**: T5 (Small, Base), BART, LongT5
- **Visualization Tools**: Matplotlib, Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/email-content-extraction.git
   cd email-content-extraction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # For Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Download pretrained models using the Hugging Face Hub:
   ```bash
   from transformers import AutoTokenizer, AutoModel
   tokenizer = AutoTokenizer.from_pretrained("t5-small")
   model = AutoModel.from_pretrained("t5-small")
   ```

## Usage
1. Preprocess the dataset:
   ```bash
   python preprocess.py --input data/raw_emails.json --output data/cleaned_emails.json
   ```
2. Perform clustering:
   ```bash
   python cluster.py --input data/cleaned_emails.json --output data/clusters.json
   ```
3. Run extractive summarization:
   ```bash
   python extractive_summarizer.py --input data/clusters.json --output data/extractive_summaries.json
   ```
4. Fine-tune and run abstractive summarization:
   ```bash
   python fine_tune.py --model t5-small --input data/extractive_summaries.json --output data/abstractive_summaries.json
   ```

## Results
- **Extractive Summarization**:
  - ROUGE-1: 35% overlap with reference summaries.
  - TextRank yielded 28% overlap.
- **Abstractive Summarization**:
  - T5-Small: Moderate performance with ROUGE-1 ~4.27.
  - BART: Achieved ROUGE-1: 48.8 with augmented data.
  - LongT5: ROUGE-1: 46.3, efficient for long text sequences.

## Future Enhancements
- **Dataset Expansion**: Incorporate more diverse email types.
- **Advanced Chunking**: Reduce context loss for large emails.
- **Custom Models**: Develop specialized models for email summarization.
- **New Metrics**: Explore metrics beyond ROUGE and BERTScores.
