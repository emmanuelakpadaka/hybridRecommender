# CPU-Optimized Hybrid Recommender System

A modular, CPU-optimized hybrid recommendation system that combines collaborative filtering, content-based filtering, and sequential patterns.

## Features

- **CPU-optimized**: Designed to run efficiently without GPU requirements
- **Memory-efficient**: Processes data in chunks for low memory usage
- **Multi-threaded**: Leverages parallel processing for performance
- **Hybrid approach**: Combines multiple recommendation techniques
- **Modular architecture**: Easy to extend and maintain

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cpu-hybrid-recommender.git
cd cpu-hybrid-recommender

# Install basic requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Optional Dependencies

For faster nearest neighbor search, install one of these packages:

```bash
# Option 1: Spotify's Annoy library
pip install annoy

# Option 2: Non-Metric Space Library (typically faster)
pip install nmslib

# Or install all optional dependencies
pip install -e ".[all]"
```

## Usage

### Using as a Command-Line Tool

```bash
# Full training mode
python -m src.train --events data/events.csv \
                   --properties1 data/item_properties_part1.csv \
                   --properties2 data/item_properties_part2.csv \
                   --categories data/category_tree.csv \
                   --output ./models \
                   --factors 100 \
                   --iterations 15

# Simple mode (faster, fewer features)
python -m src.train --events data/events.csv --simple
```

### Using as a Python Library

```python
from src.data.loaders import load_all_data
from src.models.hybrid_recommender import CPUHybridRecommender
from src.evaluation.metrics import evaluate_model, generate_recommendations

# Load data
events_df, item_props_df, category_df = load_all_data(
    'data/events.csv',
    'data/item_properties_part1.csv',
    'data/item_properties_part2.csv',
    'data/category_tree.csv'
)

# Initialize and train the model
model = CPUHybridRecommender(
    factors=100,
    iterations=15,
    output_path='./models'
)
model.train_hybrid_model(events_df, item_props_df, category_df)

# Make recommendations for a user
recommendations = model.recommend_for_user(user_id=12345, n=10)
print(recommendations)

# Find similar items
similar_items = model.get_similar_items(item_id=67890, n=10, method='hybrid')
print(similar_items)

# Save the model
model.save_model()

# Load a saved model
loaded_model = CPUHybridRecommender.load_model('./models')
```

## Data Format

The system expects the following data formats:

### Events Data

CSV file with user-item interactions:

```
visitorid,itemid,event,timestamp
12345,67890,view,1612345678
12345,67891,addtocart,1612345980
12345,67892,transaction,1612346100
```

### Item Properties Data

CSV file with item metadata:

```
itemid,property,value,timestamp
67890,categoryid,123,1612345000
67890,brand,example,1612345000
67891,categoryid,456,1612345000
```

### Category Tree Data

CSV file with category hierarchy:

```
categoryid,parentid
123,10
456,20
10,1
20,1
```

## Project Structure

```
project/
├── setup.py                 # Package installation
├── requirements.txt         # Dependencies
├── README.md                # This file
└── src/                     # Source code
    ├── __init__.py
    ├── config.py            # Global configuration
    ├── data/
    │   ├── __init__.py
    │   ├── loaders.py       # Data loading functions
    │   └── preprocessing.py # Data preprocessing
    ├── models/
    │   ├── __init__.py
    │   └── hybrid_recommender.py  # Main model class
    ├── utils/
    │   ├── __init__.py
    │   └── helpers.py       # Helper functions
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py       # Evaluation metrics
    └── train.py             # Main training script
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project incorporates several excellent open-source libraries:
- [implicit](https://github.com/benfred/implicit) for ALS collaborative filtering
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [Annoy](https://github.com/spotify/annoy) and [NMSLIB](https://github.com/nmslib/nmslib) for approximate nearest neighbors
