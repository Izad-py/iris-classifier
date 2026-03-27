# iris_classifier (Decision Tree)

## Overview
This project is an end‑to‑end machine learning example from the Digital Marketing Mastery module.  
It builds a Decision Tree classifier on the classic Iris dataset using scikit‑learn.  
The project demonstrates a full ML workflow: data loading, model training, evaluation, and saving outputs programmatically.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/iris-classifier.git
cd iris-classifier

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run the training script
python src/train.py --test-size 0.2 --random-state 42
