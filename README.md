# 📱 Mobile Product Sentiment Analyzer

A comprehensive sentiment analysis tool that analyzes mobile product reviews using Natural Language Processing (NLP) techniques and provides detailed visualizations of customer sentiment patterns.

## 🎯 Project Overview

This project performs in-depth sentiment analysis on mobile product reviews, categorizing feedback into different aspects (battery, camera, performance, etc.) and providing sentiment scores along with visualization of the results.

## ✨ Features

### Data Processing 🔄
- Product name and specifications extraction
- Rating normalization
- Review categorization by aspects (battery, camera, design, etc.)

### NLP Pipeline 🔍
- Text preprocessing
- Stop words removal
- Lemmatization
- Parts of Speech (POS) tagging
- Named Entity Recognition (NER)
- Sentiment scoring using VADER

### Visualization 📊
- Product-wise sentiment distribution
- Rating analysis
- Category-wise sentiment patterns
- Comparative analysis of ratings vs sentiment

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
spacy
```

## 📂 Project Structure

```
├── src/
│   ├── sentiment_analyzer.py
│   ├── data_processor.py
│   └── visualizer.py
├── data/
│   └── Reviews.csv
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mobile-product-sentiment-analyzer.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download(['stopwords', 'vader_lexicon', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
```

4. Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## 💻 Usage

```python
# Import required modules
from sentiment_analyzer import analyze_sentiment
from data_processor import process_reviews
from visualizer import plot_sentiments

# Load and process data
df = pd.read_csv('data/Reviews.csv')
processed_df = process_reviews(df)

# Analyze sentiments
sentiment_scores = analyze_sentiment(processed_df)

# Visualize results
plot_sentiments(sentiment_scores)
```

## 📊 Analysis Features

1. **Product Analysis**
   - Individual product sentiment scores
   - Rating distribution
   - Review categorization

2. **Category Analysis**
   - Battery feedback
   - Camera reviews
   - Performance analysis
   - Design feedback
   - Software reviews

3. **Sentiment Metrics**
   - Compound scores
   - Comparative analysis
   - Rating correlation

## 📈 Visualization Types

- Bar plots for rating distribution
- Sentiment score histograms
- Category-wise sentiment analysis
- Product-wise comparison charts
- Mean sentiment vs rating comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Tech Stack

- Python 🐍
- NLTK 📚
- spaCy 🔍
- Pandas 🐼
- Matplotlib 📊
- Seaborn 📈

## ✨ Author

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- NLTK documentation
- spaCy documentation
- VADER Sentiment Analysis tool
