# Identifying Malicious URLs

## Info

This project uses deep learning to classify a URL as malicious or benign. It includes URL encoding code, training notebooks, a saved Keras model, and a simple prediction script.

## About

The project converts URL text into a fixed-size numeric representation and feeds it to a trained model for classification. The repository also includes the dataset and notebooks used for training and visualization.

## Project Structure

```bash
Identifying_Malicious_URLs/
├── prediction.py
├── requirements.txt
├── saved Model/
│   └── malicious_url_checker_model.keras
└── src/
|    ├── Dataset/
|    │   └── malicious_phish.csv
|    ├── encoding/
|    │   └── unicode_encoding.py
|    ├── model/
|    │   └── malicious_url_model1.ipynb
|    └── visualization/
|        └── data_visulization.ipynb
|
|____ app.py
```

## How To Run

1. Clone the repository:

```bash
git clone https://github.com/Faizanfarhad/Identifying_Malicious_URLs.git
cd Identifying_Malicious_URLs
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open `prediction.py` and update the URL in this line:

```python
url = ['"g00gle.com"']
```

4. Run the prediction script:

```bash
python prediction.py
```

## Model Accuracy

0.9505 
