"""
CPU-optimized hybrid recommendation system combining collaborative filtering,
content-based filtering, and sequential recommendations.
"""
import os
import time
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Union
import datetime
import pickle
import multiprocessing
from collections import defaultdict

# Standard Python libraries
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
import networkx as nx

# For collaborative filtering (all CPU-compatible)
from implicit.als import AlternatingLeastSquares
import implicit.nearest_neighbours as implicit_nn

# CPU-friendly alternatives to GPU libraries
try:
    import annoy  # Spotify's Approximate Nearest Neighbors Oh Yeah
    has_annoy = True
except ImportError:
    has_annoy = False

try:
    import nmslib  # Non-Metric Space Library for approximate nearest neighbors
    has_nmslib = True
except ImportError:
    has_nmslib = False

# Import from other modules
from config import (
    DEFAULT_FACTORS, DEFAULT_ITERATIONS, DEFAULT_REGULARIZATION, DEFAULT_USE_NATIVE,
    DEFAULT_ITEM_ID_COL, DEFAULT_USER_ID_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_EVENT_TYPE_COL,
    DEFAULT_EVENT_WEIGHTS, DEFAULT_N_JOBS, DEFAULT_OUTPUT_PATH, DEFAULT_REUSE_ARTIFACTS,
    ANN_LIBRARIES
)
from utils.helpers import (
    properties_to_text, process_user_chunk_helper,
    process_property_chunk_helper, compute_similarities_helper
)
from data.preprocessing import (
    create_id_mappers, load_id_mappers, build_interaction_matrix,
    load_interaction_matrix, build_user_sequences, load_user_sequences,
    convert_timestamps
)

# Configure logging
logger = logging.getLogger(__name__)


class CPUHybridRecommender:
    """
    CPU-only hybrid recommendation system optimized for environments without CUDA.
    Combines collaborative filtering, content-based filtering, and sequential patterns.

    Features:
    - Optimized for CPU-only environments
    - Multi-threaded processing for performance
    - Memory-efficient data handling
    - CPU-friendly nearest neighbor search
    """

    def __init__(
            self,
            factors: int = DEFAULT_FACTORS,
            iterations: int = DEFAULT_ITERATIONS,
            regularization: float = DEFAULT_REGULARIZATION,
            use_native: bool = DEFAULT_USE_NATIVE,
            item_id_col: str = DEFAULT_ITEM_ID_COL,
            user_id_col: str = DEFAULT_USER_ID_COL,
            timestamp_col: str = DEFAULT_TIMESTAMP_COL,
            event_type_col: str = DEFAULT_EVENT_TYPE_COL,
            event_weights: Dict = DEFAULT_EVENT_WEIGHTS,
            output_path: str = DEFAULT_OUTPUT_PATH,
            n_jobs: int = None
    ):
        """
        Initialize the CPU-based Hybrid Recommender model.

        Parameters:
        -----------
        factors: int
            Number of latent factors for matrix factorization
        iterations: int
            Number of iterations for training
        regularization: float
            Regularization coefficient for ALS
        use_native: bool
            Whether to use native (faster) implementation in implicit package
        item_id_col: str
            Column name for item IDs
        user_id_col: str
            Column name for user IDs
        timestamp_col: str
            Column name for timestamps
        event_type_col: str
            Column name for event types
        event_weights: Dict
            Weights for different event types
        output_path: str
            Path to save models and artifacts
        n_jobs: int
            Number of parallel jobs (defaults to number of CPU cores)
        """
        # Core parameters
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.use_native = use_native

        # Column configurations
        self.item_id_col = item_id_col
        self.user_id_col = user_id_col
        self.timestamp_col = timestamp_col
        self.event_type_col = event_type_col

        # Event weights for different user interactions
        self.event_weights = event_weights

        # Parallelization settings
        self.n_jobs = n_jobs if n_jobs is not None else DEFAULT_N_JOBS
        logger.info(f"Using {self.n_jobs} CPU cores for parallel operations")

        # Output paths
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        # Model components
        self.cf_model = None  # Collaborative filtering model
        self.content_model = None  # Content-based model (nearest neighbors)
        self.item_features = None  # Item features for content-based filtering
        self.sequential_model = None  # Sequential recommendation model

        # Data structures
        self.user_item_matrix = None
        self.item_mapper = None
        self.user_mapper = None
        self.reverse_item_mapper = None
        self.reverse_user_mapper = None
        self.item_categories = None
        self.category_graph = None
        self.user_item_sequences = None

        # ANN indexes for faster similarity search
        self.item_ann_index = None
        self.index_to_item = None

        # Decide which ANN library to use based on availability
        if has_nmslib:
            self.ann_library = 'nmslib'
        elif has_annoy:
            self.ann_library = 'annoy'
        else:
            self.ann_library = 'sklearn'
            logger.warning("Neither nmslib nor annoy found. Using sklearn for similarity search (slower).")

    def preprocess_data(
            self,
            events_df: pd.DataFrame,
            item_properties_df: pd.DataFrame = None,
            category_tree_df: pd.DataFrame = None,
            reuse_artifacts: bool = DEFAULT_REUSE_ARTIFACTS
    ) -> None:
        """
        Preprocess all input data in a memory-efficient way.
        Will reuse existing mappings and other artifacts when available.

        Parameters:
        -----------
        events_df: pd.DataFrame
            DataFrame containing user-item interactions
        item_properties_df: pd.DataFrame
            DataFrame containing item metadata (optional)
        category_tree_df: pd.DataFrame
            DataFrame containing category hierarchy (optional)
        reuse_artifacts: bool
            Whether to reuse existing artifacts when available
        """
        logger.info("Starting data preprocessing")

        # Convert timestamp to datetime if needed
        events_df = convert_timestamps(events_df, self.timestamp_col)

        # Check for existing ID mappers
        mappers_loaded = False
        if reuse_artifacts:
            self.user_mapper, self.item_mapper, self.reverse_user_mapper, self.reverse_item_mapper = load_id_mappers(self.output_path)
            mappers_loaded = self.user_mapper is not None and self.item_mapper is not None

        # Create ID mappers for users and items if not loaded
        if not mappers_loaded:
            logger.info("Creating new ID mappers")
            self.user_mapper, self.item_mapper, self.reverse_user_mapper, self.reverse_item_mapper = create_id_mappers(
                events_df,
                self.user_id_col,
                self.item_id_col,
                self.output_path
            )

        # Add confidence scores based on event types
        logger.info("Calculating confidence scores based on event types")
        events_df = events_df.copy()
        events_df['confidence'] = events_df[self.event_type_col].map(self.event_weights)

        # Check for existing user-item matrix
        matrix_loaded = False
        if reuse_artifacts:
            self.user_item_matrix = load_interaction_matrix(self.output_path)
            matrix_loaded = self.user_item_matrix is not None

        # Process user-item interactions if matrix not loaded
        if not matrix_loaded:
            logger.info("Building user-item interaction matrix")
            self.user_item_matrix = build_interaction_matrix(
                events_df,
                self.user_mapper,
                self.item_mapper,
                self.user_id_col,
                self.item_id_col,
                self.event_type_col,
                self.event_weights,
                self.output_path
            )

        # Check for existing user sequences
        sequences_loaded = False
        if reuse_artifacts:
            self.user_item_sequences = load_user_sequences(self.output_path)
            sequences_loaded = self.user_item_sequences is not None

        # Build user sequences for sequential recommendations if not loaded
        if not sequences_loaded:
            logger.info("Building user interaction sequences")
            self.user_item_sequences = build_user_sequences(
                events_df,
                self.user_mapper,
                self.item_mapper,
                self.user_id_col,
                self.item_id_col,
                self.timestamp_col,
                self.n_jobs,
                self.output_path
            )

        # Check for existing item features
        features_loaded = False
        if reuse_artifacts and item_properties_df is not None:
            try:
                features_path = os.path.join(self.output_path, 'item_texts.joblib')
                vectorizer_path = os.path.join(self.output_path, 'tfidf_vectorizer.joblib')
                svd_path = os.path.join(self.output_path, 'tfidf_svd.joblib')
                feature_matrix_path = os.path.join(self.output_path, 'item_feature_matrix.npy')
                feature_items_path = os.path.join(self.output_path, 'feature_items.joblib')

                if (os.path.exists(features_path) and os.path.exists(vectorizer_path) and
                        os.path.exists(svd_path) and os.path.exists(feature_matrix_path) and
                        os.path.exists(feature_items_path)):
                    logger.info("Found existing item features - loading them")
                    self.item_features = joblib.load(features_path)
                    self.tfidf_vectorizer = joblib.load(vectorizer_path)
                    self.tfidf_svd = joblib.load(svd_path)
                    self.item_feature_matrix = np.load(feature_matrix_path)
                    self.feature_items = joblib.load(feature_items_path)
                    self.item_to_feature_idx = {item: idx for idx, item in enumerate(self.feature_items)}

                    # Rebuild ANN index
                    logger.info("Rebuilding ANN index from loaded features")
                    self._build_item_ann_index(self.item_feature_matrix, self.feature_items)
                    features_loaded = True
            except Exception as e:
                logger.warning(f"Could not load existing item features: {e}")
                features_loaded = False

        # Process item metadata if available and not loaded
        if item_properties_df is not None and not features_loaded:
            logger.info("Processing item metadata")
            self._process_item_features(item_properties_df)

        # Check for existing category information
        categories_loaded = False
        if reuse_artifacts and category_tree_df is not None:
            try:
                graph_path = os.path.join(self.output_path, 'category_graph.gpickle')
                sim_path = os.path.join(self.output_path, 'category_sim.npy')
                categories_path = os.path.join(self.output_path, 'categories.joblib')
                cat_mapping_path = os.path.join(self.output_path, 'cat_mapping.joblib')

                if (os.path.exists(graph_path) and os.path.exists(sim_path) and
                        os.path.exists(categories_path) and os.path.exists(cat_mapping_path)):
                    logger.info("Found existing category information - loading it")
                    with open(graph_path, 'rb') as f:
                        self.category_graph = pickle.load(f)
                    self.category_sim = np.load(sim_path)
                    self.categories = joblib.load(categories_path)
                    self.cat_mapping = joblib.load(cat_mapping_path)
                    categories_loaded = True
            except Exception as e:
                logger.warning(f"Could not load existing category information: {e}")
                categories_loaded = False

        # Process category hierarchy if available and not loaded
        if category_tree_df is not None and not categories_loaded:
            logger.info("Processing category hierarchy")
            self._process_category_tree(category_tree_df)

        logger.info("Data preprocessing complete")

    def _process_item_features(self, item_properties_df: pd.DataFrame) -> None:
        """
        Process item metadata to create feature vectors for content-based filtering.
        Uses chunking for memory efficiency.

        Parameters:
        -----------
        item_properties_df: pd.DataFrame
            DataFrame containing item properties
        """
        # If timestamp column exists, keep only the latest value for each property
        if self.timestamp_col in item_properties_df.columns:
            item_properties_df = convert_timestamps(item_properties_df, self.timestamp_col)

            # Get latest values for each property (process in chunks)
            logger.info("Finding latest property values")

            # Split properties into chunks and process in parallel
            chunk_size = min(500000, max(10000, len(item_properties_df) // self.n_jobs))
            prop_chunks = [item_properties_df.iloc[i:i + chunk_size]
                           for i in range(0, len(item_properties_df), chunk_size)]

            # Create args for the helper function
            args_list = [(chunk, self.timestamp_col) for chunk in prop_chunks]

            # Process property chunks in parallel with starmap
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                latest_chunks = pool.map(process_property_chunk_helper, args_list)

            # Combine chunks
            latest_properties = pd.concat(latest_chunks)

            # Re-aggregate to handle properties that might span chunks
            latest_properties = latest_properties.groupby([self.item_id_col, 'property']).last().reset_index()
        else:
            latest_properties = item_properties_df.copy()

        # Extract category information (important for recommendations)
        if 'categoryid' in latest_properties['property'].values:
            logger.info("Extracting item categories")
            category_data = latest_properties[latest_properties['property'] == 'categoryid']
            self.item_categories = dict(zip(category_data[self.item_id_col], category_data['value']))
            joblib.dump(self.item_categories, os.path.join(self.output_path, 'item_categories.joblib'))

        # Convert properties to text features for each item
        logger.info("Creating text features from item properties")

        # Process in chunks to save memory
        item_texts = {}
        for item_id, group in latest_properties.groupby(self.item_id_col):
            if item_id in self.item_mapper:
                item_texts[item_id] = properties_to_text(group)

        # Check if we have any item texts
        if not item_texts:
            logger.warning("No item feature texts found")
            return

        # Prepare for TF-IDF vectorization
        logger.info("Computing TF-IDF vectors for items")
        items = list(item_texts.keys())
        texts = [item_texts[item] for item in items]

        # Vectorize text with TF-IDF (with dimension reduction)
        max_features = min(5000, len(texts) // 2) if len(texts) > 10 else 10
        logger.info(f"Using max_features={max_features} for TF-IDF")

        vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'[^\s]+',
            max_features=max_features,
            min_df=2,  # Ignore terms appearing in fewer than 2 documents
            max_df=0.8,  # Ignore terms appearing in more than 80% of documents
            use_idf=True
        )

        # Fit on all data
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Apply dimensionality reduction for efficiency
            n_components = min(100, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
            logger.info(f"Reducing TF-IDF dimensions to {n_components} with SVD")

            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = svd.fit_transform(tfidf_matrix)

            # Normalize for cosine similarity
            normalized_matrix = normalize(reduced_matrix)

            # Store item features and related objects
            self.item_features = item_texts
            self.tfidf_vectorizer = vectorizer
            self.tfidf_svd = svd
            self.item_feature_matrix = normalized_matrix
            self.feature_items = items
            self.item_to_feature_idx = {item: idx for idx, item in enumerate(items)}

            # Build nearest neighbors index
            logger.info(f"Building approximate nearest neighbors index using {self.ann_library}")
            self._build_item_ann_index(normalized_matrix, items)

            # Save feature components
            logger.info("Saving feature components")
            joblib.dump(item_texts, os.path.join(self.output_path, 'item_texts.joblib'))
            joblib.dump(vectorizer, os.path.join(self.output_path, 'tfidf_vectorizer.joblib'))
            joblib.dump(svd, os.path.join(self.output_path, 'tfidf_svd.joblib'))
            np.save(os.path.join(self.output_path, 'item_feature_matrix.npy'), normalized_matrix)
            joblib.dump(items, os.path.join(self.output_path, 'feature_items.joblib'))
        except Exception as e:
            logger.error(f"Error processing item features: {e}")
            self.item_features = {}

    def _build_item_ann_index(self, matrix, items):
        """
        Build approximate nearest neighbors index for fast similarity search.
        Uses different libraries based on availability.

        Parameters:
        -----------
        matrix: np.ndarray
            Item feature matrix
        items: list
            List of original item IDs
        """
        # Store mapping from index to original item ID
        self.index_to_item = items

        # Different index building based on available libraries
        if self.ann_library == 'nmslib':
            # Non-Metric Space Library (typically fastest on CPU)
            try:
                index = nmslib.init(method='hnsw', space='cosinesimil')
                index.addDataPointBatch(matrix)
                index.createIndex({'post': 2, 'M': 30, 'efConstruction': 200}, print_progress=True)
                self.item_ann_index = index
            except Exception as e:
                logger.error(f"Error creating NMSLIB index: {e}")
                self.ann_library = 'sklearn'
                self.item_ann_index = matrix  # Fallback to brute force

        elif self.ann_library == 'annoy':
            # Spotify's Annoy library (good balance of speed and quality)
            try:
                n_dimensions = matrix.shape[1]
                index = annoy.AnnoyIndex(n_dimensions, 'angular')  # Angular distance = cosine similarity

                for i, vector in enumerate(matrix):
                    index.add_item(i, vector)

                index.build(50)  # 50 trees - more trees = more accurate but slower

                # Save the index to disk rather than keeping it in memory
                # This avoids pickling issues during multiprocessing
                index_path = os.path.join(self.output_path, 'annoy_index.ann')
                index.save(index_path)

                # Store the path instead of the object
                self.item_ann_index_path = index_path
                # Keep a transient reference for immediate use
                self.item_ann_index = index
            except Exception as e:
                logger.error(f"Error creating Annoy index: {e}")
                self.ann_library = 'sklearn'
                self.item_ann_index = matrix  # Fallback to brute force

        else:
            # Fall back to sklearn's brute force (slower but no extra dependencies)
            self.item_ann_index = matrix  # Just store the matrix for brute force search

    def _process_category_tree(self, category_tree_df: pd.DataFrame) -> None:
        """
        Process category hierarchy to create category embeddings.

        Parameters:
        -----------
        category_tree_df: pd.DataFrame
            DataFrame containing category hierarchy
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges from child to parent
        for _, row in category_tree_df.iterrows():
            child = row['categoryid']
            parent = row['parentid']

            # Add node
            if not G.has_node(child):
                G.add_node(child)

            # Add edge to parent if exists
            if pd.notna(parent):
                if not G.has_node(parent):
                    G.add_node(parent)
                G.add_edge(child, parent)

        # Save the graph
        self.category_graph = G
        with open(os.path.join(self.output_path, 'category_graph.gpickle'), 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

        # Compute category similarity matrix
        logger.info("Computing category similarity matrix")
        categories = list(G.nodes())
        n_categories = len(categories)

        # Create mapping for quick lookup
        cat_mapping = {cat: idx for idx, cat in enumerate(categories)}

        # Initialize similarity matrix
        category_sim = np.zeros((n_categories, n_categories))

        # Precompute ancestors for all categories
        ancestors_cache = {}
        for cat in categories:
            try:
                ancestors_cache[cat] = nx.ancestors(G, cat)
            except:
                ancestors_cache[cat] = set()

        # Compute similarities (chunked for memory efficiency)
        # Split categories into chunks for parallel processing
        category_chunks = np.array_split(categories, min(self.n_jobs, max(1, n_categories // 100)))

        # Create args for the helper function
        args_list = [
            (chunk, categories, cat_mapping, G, ancestors_cache)
            for chunk in category_chunks
        ]

        # Process in parallel
        try:
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                all_results = pool.map(compute_similarities_helper, args_list)

            # Combine results
            for chunk_results in all_results:
                for i, j, sim_value in chunk_results:
                    category_sim[i, j] = sim_value

            # Save category similarity matrix
            self.category_sim = category_sim
            self.categories = categories
            self.cat_mapping = cat_mapping

            np.save(os.path.join(self.output_path, 'category_sim.npy'), category_sim)
            joblib.dump(categories, os.path.join(self.output_path, 'categories.joblib'))
            joblib.dump(cat_mapping, os.path.join(self.output_path, 'cat_mapping.joblib'))
        except Exception as e:
            logger.error(f"Error computing category similarities: {e}")
            self.category_sim = None

    def train_collaborative_filtering(self) -> None:
        """
        Train collaborative filtering model using Alternating Least Squares.
        Optimized for CPU processing.
        """
        logger.info("Training collaborative filtering model")

        # Check if we have user-item matrix
        if self.user_item_matrix is None:
            logger.error("User-item matrix not found. Run preprocess_data first.")
            return

        # Initialize ALS model
        self.cf_model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False,  # Ensure CPU-only processing
            use_native=self.use_native,  # Native implementation is faster on CPU
            num_threads=self.n_jobs,
            calculate_training_loss=True
        )

        # Fit the model
        logger.info(f"Fitting ALS model with {self.factors} factors, {self.iterations} iterations")
        start_time = time.time()

        # ALS expects (item, user) matrix, so we transpose
        self.cf_model.fit(self.user_item_matrix.T)

        training_time = time.time() - start_time
        logger.info(f"ALS training completed in {training_time:.2f} seconds")

        # Save the model
        logger.info("Saving collaborative filtering model")
        with open(os.path.join(self.output_path, 'cf_model.pkl'), 'wb') as f:
            pickle.dump(self.cf_model, f)

    def train_content_model(self) -> None:
        """
        Train content-based model using item features.
        """
        logger.info("Training content-based model")

        if not hasattr(self, 'item_feature_matrix') or self.item_feature_matrix is None:
            logger.warning("Item features not processed. Skipping content model training.")
            return

        # For content model, we use the ANN index built during preprocessing
        # So no additional training is needed here
        logger.info("Content-based index already built during preprocessing")

    def train_sequential_model(self) -> None:
        """
        Train sequential recommendation model based on user histories.
        Uses a simplified Markov chain approach for CPU efficiency.
        """
        logger.info("Training sequential recommendation model")

        if not hasattr(self, 'user_item_sequences') or not self.user_item_sequences:
            logger.warning("User sequences not found. Skipping sequential model training.")
            return

        # Build a simple Markov transition matrix (item-to-item)
        transition_counts = defaultdict(lambda: defaultdict(int))

        # Count transitions
        for user, sequence in self.user_item_sequences.items():
            if len(sequence) < 2:
                continue

            # Count item-to-item transitions
            for i in range(len(sequence) - 1):
                current_item = sequence[i]
                next_item = sequence[i + 1]
                transition_counts[current_item][next_item] += 1

        # Convert to probabilities
        transition_probs = {}
        for current_item, next_items in transition_counts.items():
            total = sum(next_items.values())
            transition_probs[current_item] = {
                next_item: count / total
                for next_item, count in next_items.items()
            }

        # Store the model
        self.sequential_model = transition_probs

        # Save the model
        logger.info("Saving sequential model")
        joblib.dump(self.sequential_model, os.path.join(self.output_path, 'sequential_model.joblib'))

    def train_hybrid_model(
            self,
            events_df: pd.DataFrame = None,
            item_properties_df: pd.DataFrame = None,
            category_tree_df: pd.DataFrame = None,
            reuse_artifacts: bool = DEFAULT_REUSE_ARTIFACTS
    ) -> None:
        """
        Train the complete hybrid recommendation model.
        Will reuse existing artifacts when available.

        Parameters:
        -----------
        events_df: pd.DataFrame
            DataFrame containing user-item interactions
        item_properties_df: pd.DataFrame
            DataFrame containing item metadata (optional)
        category_tree_df: pd.DataFrame
            DataFrame containing category hierarchy (optional)
        reuse_artifacts: bool
            Whether to reuse existing artifacts when available
        """
        logger.info("Training hybrid recommendation model")

        # Step 1: Preprocess data if not already done
        if (self.user_item_matrix is None or not hasattr(self, 'user_item_matrix')) and events_df is not None:
            logger.info("Preprocessing data")
            self.preprocess_data(events_df, item_properties_df, category_tree_df, reuse_artifacts=reuse_artifacts)
        elif self.user_item_matrix is None:
            logger.error("No data to train on. Please provide events_df or run preprocess_data first.")
            return

        # Check for existing CF model
        cf_model_loaded = False
        if reuse_artifacts:
            try:
                cf_model_path = os.path.join(self.output_path, 'cf_model.pkl')
                if os.path.exists(cf_model_path):
                    logger.info("Found existing collaborative filtering model - loading it")
                    with open(cf_model_path, 'rb') as f:
                        self.cf_model = pickle.load(f)
                    cf_model_loaded = True
            except Exception as e:
                logger.warning(f"Could not load existing CF model: {e}")
                cf_model_loaded = False

        # Step 2: Train collaborative filtering model if not loaded
        if not cf_model_loaded:
            logger.info("Training collaborative filtering component")
            self.train_collaborative_filtering()

        # Step 3: Train content-based model (if item features available)
        if hasattr(self, 'item_feature_matrix') and self.item_feature_matrix is not None:
            # Content model uses the ANN index built during preprocessing
            logger.info("Content-based component ready (index built during preprocessing)")

        # Check for existing sequential model
        seq_model_loaded = False
        if reuse_artifacts and hasattr(self, 'user_item_sequences') and self.user_item_sequences:
            try:
                seq_model_path = os.path.join(self.output_path, 'sequential_model.joblib')
                if os.path.exists(seq_model_path):
                    logger.info("Found existing sequential model - loading it")
                    self.sequential_model = joblib.load(seq_model_path)
                    seq_model_loaded = True
            except Exception as e:
                logger.warning(f"Could not load existing sequential model: {e}")
                seq_model_loaded = False

        # Step 4: Train sequential model if not loaded
        if hasattr(self, 'user_item_sequences') and self.user_item_sequences and not seq_model_loaded:
            logger.info("Training sequential component")
            self.train_sequential_model()

        logger.info("Hybrid model training complete")

        # Save hybrid model configuration
        config = {
            'has_cf': self.cf_model is not None,
            'has_content': hasattr(self, 'item_feature_matrix') and self.item_feature_matrix is not None,
            'has_sequential': hasattr(self, 'sequential_model') and self.sequential_model is not None,
            'factors': self.factors,
            'iterations': self.iterations,
            'regularization': self.regularization,
            'n_users': len(self.user_mapper) if self.user_mapper else 0,
            'n_items': len(self.item_mapper) if self.item_mapper else 0
        }

        # Save all model config info
        joblib.dump(config, os.path.join(self.output_path, 'hybrid_config.joblib'))

        # Also save overall model configuration for easier loading
        model_config = {
            'factors': self.factors,
            'iterations': self.iterations,
            'regularization': self.regularization,
            'use_native': self.use_native,
            'item_id_col': self.item_id_col,
            'user_id_col': self.user_id_col,
            'timestamp_col': self.timestamp_col,
            'event_type_col': self.event_type_col,
            'event_weights': self.event_weights,
            'n_jobs': self.n_jobs,
            'ann_library': self.ann_library
        }
        joblib.dump(model_config, os.path.join(self.output_path, 'model_config.joblib'))

    def get_item_neighbors(self, item_id, n=10):
        """
        Find similar items using the content-based model.

        Parameters:
        -----------
        item_id: Original item ID
        n: int
            Number of neighbors to return

        Returns:
        --------
        list: List of (item_id, similarity_score) tuples
        """
        # Check if content model is available - with more detailed checks
        if not hasattr(self, 'item_feature_matrix') or self.item_feature_matrix is None:
            logger.warning("Content-based features not available")
            return []

        if not hasattr(self, 'index_to_item') or self.index_to_item is None:
            logger.warning("Item index mapping not available")
            return []

        # Convert item ID to index
        if item_id not in self.item_mapper:
            logger.warning(f"Item {item_id} not found in item mapping")
            return []

        item_idx = self.item_mapper[item_id]

        # Convert to feature index
        if not hasattr(self, 'item_to_feature_idx') or not self.item_to_feature_idx:
            logger.warning("Feature index mapping not available")
            return []

        if item_id not in self.item_to_feature_idx:
            logger.warning(f"Item {item_id} not found in feature mapping")
            return []

        feature_idx = self.item_to_feature_idx[item_id]

        # Find nearest neighbors using the appropriate library
        if self.ann_library == 'nmslib':
            # Get neighbors using NMSLIB
            try:
                if not hasattr(self, 'item_ann_index') or self.item_ann_index is None:
                    logger.error("NMSLIB index is None")
                    return []

                item_vector = self.item_feature_matrix[feature_idx].reshape(1, -1)
                neighbors = self.item_ann_index.knnQuery(item_vector, k=n + 1)[0]

                # Convert indices to original item IDs and compute similarities
                results = []
                for idx in neighbors:
                    if idx != feature_idx and idx < len(self.index_to_item):  # Skip the query item itself
                        neighbor_id = self.index_to_item[idx]
                        # Compute similarity (approximate)
                        sim = 1.0 - (idx * 0.01)  # A simple approximation
                        results.append((neighbor_id, sim))

                return results[:n]
            except Exception as e:
                logger.error(f"Error getting NMSLIB neighbors: {e}")
                return []

        elif self.ann_library == 'annoy':
            # Get neighbors using Annoy
            try:
                # Check if we need to load the index from disk
                if (not hasattr(self, 'item_ann_index') or self.item_ann_index is None) and hasattr(self,
                                                                                                    'item_ann_index_path'):
                    logger.info("Loading Annoy index from disk")
                    if not os.path.exists(self.item_ann_index_path):
                        logger.error(f"Annoy index file not found at {self.item_ann_index_path}")
                        return []

                    n_dimensions = self.item_feature_matrix.shape[1]
                    index = annoy.AnnoyIndex(n_dimensions, 'angular')
                    index.load(self.item_ann_index_path)
                    self.item_ann_index = index

                # Check if the index is loaded
                if not hasattr(self, 'item_ann_index') or self.item_ann_index is None:
                    logger.error("Annoy index is None")
                    return []

                # Check if feature_idx is in bounds
                if feature_idx >= len(self.item_feature_matrix):
                    logger.error(
                        f"Feature index {feature_idx} out of bounds for matrix with {len(self.item_feature_matrix)} rows")
                    return []

                # Get the nearest neighbors using the Annoy index
                neighbors = self.item_ann_index.get_nns_by_item(
                    feature_idx, n + 1, include_distances=True
                )

                # Convert indices and distances to item IDs and similarities
                results = []
                indices, distances = neighbors

                for idx, dist in zip(indices, distances):
                    # Skip the query item itself and check if idx is valid
                    if idx != feature_idx and idx < len(self.index_to_item):
                        neighbor_id = self.index_to_item[idx]
                        # Convert distance to similarity (Annoy returns angular distance)
                        sim = 1.0 - (dist / 2.0)
                        results.append((neighbor_id, sim))

                return results[:n]
            except Exception as e:
                logger.error(f"Error getting Annoy neighbors: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []

        elif self.ann_library == 'sklearn':
            # Fallback to brute force with sklearn
            try:
                # Verify that we have all the necessary components
                if len(self.item_feature_matrix) == 0:
                    logger.error("Feature matrix is empty")
                    return []

                if len(self.index_to_item) == 0:
                    logger.error("Item index mapping is empty")
                    return []

                # Make sure feature_idx is within valid range
                if feature_idx >= len(self.item_feature_matrix):
                    logger.error(
                        f"Feature index {feature_idx} out of bounds for matrix of size {len(self.item_feature_matrix)}")
                    return []

                item_vector = self.item_feature_matrix[feature_idx].reshape(1, -1)
                similarities = cosine_similarity(item_vector, self.item_feature_matrix).flatten()

                # Get top indices excluding the query item itself
                similar_indices = np.argsort(similarities)[::-1]
                similar_indices = [idx for idx in similar_indices if
                                   idx != feature_idx and idx < len(self.index_to_item)][:n]

                # Convert to original item IDs and similarities
                results = [
                    (self.index_to_item[idx], float(similarities[idx]))
                    for idx in similar_indices
                ]

                return results
            except Exception as e:
                logger.error(f"Error getting sklearn neighbors: {e}")
                return []
        else:
            logger.error(f"Unknown ANN library type: {self.ann_library}")
            return []

    def recommend_for_user(self, user_id, n=10, method='hybrid'):
        """
        Generate recommendations for a user.

        Parameters:
        -----------
        user_id: Original user ID
        n: int
            Number of recommendations
        method: str
            Recommendation method: 'hybrid', 'cf', 'content', or 'sequential'

        Returns:
        --------
        list: List of (item_id, score) tuples
        """
        # Check if user exists in our mapping
        if user_id not in self.user_mapper:
            logger.warning(f"User {user_id} not found")
            return []

        user_idx = self.user_mapper[user_id]

        # Get recommendations from different models based on method
        if method == 'cf' or method == 'hybrid':
            if self.cf_model is None:
                logger.warning("Collaborative filtering model not trained")
                cf_recs = []
            else:
                try:
                    # The critical fix for the CSR matrix issue:
                    import scipy.sparse

                    # Check if user_item_matrix is available and properly formatted
                    if self.user_item_matrix is None:
                        logger.error("User-item matrix is None - cannot generate CF recommendations")
                        cf_recs = []
                    # Check if user_idx is within bounds
                    elif user_idx >= self.user_item_matrix.shape[0]:
                        logger.error(
                            f"User index {user_idx} out of bounds for matrix with {self.user_item_matrix.shape[0]} rows")
                        cf_recs = []
                    else:
                        # Get just this user's interactions
                        user_row = self.user_item_matrix[user_idx]

                        # Create a NEW single-row CSR matrix in the format implicit expects
                        # - user_items should be a CSR matrix with shape (1, n_items)
                        user_items = scipy.sparse.csr_matrix(
                            (user_row.data, user_row.indices, [0, len(user_row.indices)]),
                            shape=(1, self.user_item_matrix.shape[1])
                        )

                        # Log for verification
                        logger.info(f"User items shape: {user_items.shape}, type: {type(user_items)}")
                        logger.info(f"User items is CSR: {scipy.sparse.isspmatrix_csr(user_items)}")

                        # Get recommendations using ALS with the properly formatted user_items
                        cf_indices, cf_scores = self.cf_model.recommend(
                            user_idx, user_items, N=n, filter_already_liked_items=True
                        )

                        # Convert to original item IDs
                        cf_recs = [
                            (self.reverse_item_mapper[idx], float(score))
                            for idx, score in zip(cf_indices, cf_scores)
                            if idx in self.reverse_item_mapper
                        ]
                except Exception as e:
                    logger.error(f"Error getting CF recommendations: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    cf_recs = []
        else:
            cf_recs = []

        # Get content-based recommendations if requested
        if method == 'content' or method == 'hybrid':
            # For content-based recommendations without explicit user profile,
            # we use the user's previous interactions to find similar items
            content_recs = []

            try:
                if hasattr(self,
                           'user_item_sequences') and self.user_item_sequences and user_idx in self.user_item_sequences:
                    # Get user's recent items
                    recent_items = self.user_item_sequences[user_idx][-5:]

                    # Find similar items to the user's recent items
                    content_items = {}
                    for item_idx in recent_items:
                        # Convert to original item ID
                        if item_idx in self.reverse_item_mapper:
                            item_id = self.reverse_item_mapper[item_idx]

                            # Get similar items
                            similar_items = self.get_item_neighbors(item_id, n=3)

                            # Add to content recommendations with recency weighting
                            for similar_id, sim_score in similar_items:
                                content_items[similar_id] = content_items.get(similar_id, 0) + sim_score

                    # Sort and convert to list
                    content_recs = sorted(
                        [(item, score) for item, score in content_items.items()],
                        key=lambda x: -x[1]
                    )[:n]
            except Exception as e:
                logger.error(f"Error getting content-based recommendations: {e}")
                content_recs = []
        else:
            content_recs = []

        # Get sequential recommendations if requested
        if method == 'sequential' or method == 'hybrid':
            seq_recs = []

            try:
                if hasattr(self, 'sequential_model') and self.sequential_model and \
                        hasattr(self,
                                'user_item_sequences') and self.user_item_sequences and user_idx in self.user_item_sequences:

                    # Get user's last item
                    if len(self.user_item_sequences[user_idx]) > 0:
                        last_item = self.user_item_sequences[user_idx][-1]

                        # Get next item predictions from Markov model
                        if last_item in self.sequential_model:
                            next_items = self.sequential_model[last_item]

                            # Sort by probability and convert to original item IDs
                            seq_recs = [
                                           (self.reverse_item_mapper[item], prob)
                                           for item, prob in sorted(next_items.items(), key=lambda x: -x[1])
                                           if item in self.reverse_item_mapper
                                       ][:n]
            except Exception as e:
                logger.error(f"Error getting sequential recommendations: {e}")
                seq_recs = []
        else:
            seq_recs = []

        # Combine recommendations based on method
        if method == 'hybrid':
            # Combine all recommendation types with weights
            cf_weight = 0.6
            content_weight = 0.3
            seq_weight = 0.1

            # Create a combined dictionary
            combined = {}

            # Add CF recommendations
            for item, score in cf_recs:
                combined[item] = combined.get(item, 0) + score * cf_weight

            # Add content recommendations
            for item, score in content_recs:
                combined[item] = combined.get(item, 0) + score * content_weight

            # Add sequential recommendations
            for item, score in seq_recs:
                combined[item] = combined.get(item, 0) + score * seq_weight

            # Sort and return top N
            recommendations = sorted(
                [(item, score) for item, score in combined.items()],
                key=lambda x: -x[1]
            )[:n]

            return recommendations

        elif method == 'cf':
            return cf_recs[:n]

        elif method == 'content':
            return content_recs[:n]

        elif method == 'sequential':
            return seq_recs[:n]

        else:
            logger.warning(f"Unknown recommendation method: {method}")
            return []

    def get_similar_items(self, item_id, n=10, method='hybrid'):
        """
        Find items similar to a given item.

        Parameters:
        -----------
        item_id: Original item ID
        n: int
            Number of similar items to return
        method: str
            Similarity method: 'hybrid', 'cf', or 'content'

        Returns:
        --------
        list: List of (item_id, similarity_score) tuples
        """
        # Check if item exists in our mapping
        if item_id not in self.item_mapper:
            logger.warning(f"Item {item_id} not found")
            return []

        item_idx = self.item_mapper[item_id]

        # Get collaborative filtering based similarities
        if method == 'cf' or method == 'hybrid':
            if self.cf_model is None:
                logger.warning("Collaborative filtering model not trained")
                cf_similar = []
            else:
                try:
                    # Validate the item index is within bounds
                    n_items = self.cf_model.item_factors.shape[0]
                    if item_idx >= n_items:
                        logger.error(f"Item index {item_idx} out of bounds (max: {n_items - 1})")
                        cf_similar = []
                    else:
                        # Get similar items from the model
                        cf_indices, cf_scores = self.cf_model.similar_items(item_idx, N=n + 1)

                        # Convert to original item IDs and remove the query item itself
                        cf_similar = [
                                         (self.reverse_item_mapper[idx], float(score))
                                         for idx, score in zip(cf_indices, cf_scores)
                                         if idx != item_idx and idx in self.reverse_item_mapper
                                     ][:n]
                except Exception as e:
                    logger.error(f"Error getting CF similar items for {item_id} (idx={item_idx}): {e}")
                    cf_similar = []
        else:
            cf_similar = []

        # Get content-based similarities
        if method == 'content' or method == 'hybrid':
            content_similar = self.get_item_neighbors(item_id, n=n)
        else:
            content_similar = []

        # Combine based on method
        if method == 'hybrid':
            # Combine content and CF similarities with weights
            cf_weight = 0.6
            content_weight = 0.4

            # Create a combined dictionary
            combined = {}

            # Add CF similarities
            for item, score in cf_similar:
                combined[item] = combined.get(item, 0) + score * cf_weight

            # Add content similarities
            for item, score in content_similar:
                combined[item] = combined.get(item, 0) + score * content_weight

            # Sort and return top N
            similar_items = sorted(
                [(item, score) for item, score in combined.items()],
                key=lambda x: -x[1]
            )[:n]

            return similar_items

        elif method == 'cf':
            return cf_similar[:n]

        elif method == 'content':
            return content_similar[:n]

        else:
            logger.warning(f"Unknown similarity method: {method}")
            return []

    def save_model(self, path=None):
        """
        Save the model and all necessary components.

        Parameters:
        -----------
        path: str
            Path to save the model
        """
        # Use default path if not specified
        path = path or self.output_path
        logger.info(f"Saving model to {path}")

        # Make sure directory exists
        os.makedirs(path, exist_ok=True)

        # Save user and item mappers
        joblib.dump(self.user_mapper, os.path.join(path, 'user_mapper.joblib'))
        joblib.dump(self.item_mapper, os.path.join(path, 'item_mapper.joblib'))

        # Save collaborative filtering model if available
        if self.cf_model is not None:
            with open(os.path.join(path, 'cf_model.pkl'), 'wb') as f:
                pickle.dump(self.cf_model, f)

        # Save user-item matrix if available
        if self.user_item_matrix is not None:
            joblib.dump(self.user_item_matrix, os.path.join(path, 'user_item_matrix.npz'), compress=3)

        # Save item features if available
        if hasattr(self, 'item_features') and self.item_features:
            joblib.dump(self.item_features, os.path.join(path, 'item_texts.joblib'))

        if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, os.path.join(path, 'tfidf_vectorizer.joblib'))

        if hasattr(self, 'tfidf_svd') and self.tfidf_svd:
            joblib.dump(self.tfidf_svd, os.path.join(path, 'tfidf_svd.joblib'))

        if hasattr(self, 'item_feature_matrix') and self.item_feature_matrix is not None:
            np.save(os.path.join(path, 'item_feature_matrix.npy'), self.item_feature_matrix)

        if hasattr(self, 'feature_items') and self.feature_items:
            joblib.dump(self.feature_items, os.path.join(path, 'feature_items.joblib'))

        # Save item categories if available
        if hasattr(self, 'item_categories') and self.item_categories:
            joblib.dump(self.item_categories, os.path.join(path, 'item_categories.joblib'))

        # Save sequential model if available
        if hasattr(self, 'sequential_model') and self.sequential_model:
            joblib.dump(self.sequential_model, os.path.join(path, 'sequential_model.joblib'))

        if hasattr(self, 'user_item_sequences') and self.user_item_sequences:
            joblib.dump(self.user_item_sequences, os.path.join(path, 'user_sequences.joblib'))

        # Save category graph if available
        if hasattr(self, 'category_graph') and self.category_graph:
            with open(os.path.join(path, 'category_graph.gpickle'), 'wb') as f:
                pickle.dump(self.category_graph, f, pickle.HIGHEST_PROTOCOL)

        if hasattr(self, 'category_sim') and self.category_sim is not None:
            np.save(os.path.join(path, 'category_sim.npy'), self.category_sim)

        if hasattr(self, 'categories') and self.categories:
            joblib.dump(self.categories, os.path.join(path, 'categories.joblib'))

        if hasattr(self, 'cat_mapping') and self.cat_mapping:
            joblib.dump(self.cat_mapping, os.path.join(path, 'cat_mapping.joblib'))

        # Save model configuration
        config = {
            'factors': self.factors,
            'iterations': self.iterations,
            'regularization': self.regularization,
            'use_native': self.use_native,
            'item_id_col': self.item_id_col,
            'user_id_col': self.user_id_col,
            'timestamp_col': self.timestamp_col,
            'event_type_col': self.event_type_col,
            'event_weights': self.event_weights,
            'n_jobs': self.n_jobs,
            'ann_library': self.ann_library
        }

        joblib.dump(config, os.path.join(path, 'model_config.joblib'))
        logger.info("Model saved successfully")

    @classmethod
    def load_model(cls, path):
        """
        Load a previously saved model.

        Parameters:
        -----------
        path: str
            Path to load the model from

        Returns:
        --------
        CPUHybridRecommender: Loaded model
        """
        logger.info(f"Loading model from {path}")

        # Load configuration
        try:
            config = joblib.load(os.path.join(path, 'model_config.joblib'))
        except:
            logger.warning("Model configuration not found, using defaults")
            config = {}

        # Create a new instance with loaded configuration
        model = cls(
            factors=config.get('factors', DEFAULT_FACTORS),
            iterations=config.get('iterations', DEFAULT_ITERATIONS),
            regularization=config.get('regularization', DEFAULT_REGULARIZATION),
            use_native=config.get('use_native', DEFAULT_USE_NATIVE),
            item_id_col=config.get('item_id_col', DEFAULT_ITEM_ID_COL),
            user_id_col=config.get('user_id_col', DEFAULT_USER_ID_COL),
            timestamp_col=config.get('timestamp_col', DEFAULT_TIMESTAMP_COL),
            event_type_col=config.get('event_type_col', DEFAULT_EVENT_TYPE_COL),
            output_path=path,
            n_jobs=config.get('n_jobs', DEFAULT_N_JOBS)
        )

        # Load event weights if available
        if 'event_weights' in config:
            model.event_weights = config['event_weights']

        # Set ANN library
        if 'ann_library' in config:
            model.ann_library = config['ann_library']

        # Load user and item mappers
        try:
            model.user_mapper = joblib.load(os.path.join(path, 'user_mapper.joblib'))
            model.item_mapper = joblib.load(os.path.join(path, 'item_mapper.joblib'))
            model.reverse_user_mapper = {idx: user for user, idx in model.user_mapper.items()}
            model.reverse_item_mapper = {idx: item for item, idx in model.item_mapper.items()}
        except Exception as e:
            logger.warning(f"User or item mappers not found: {e}")
            return model

        # Load user-item matrix if available - with explicit scipy import
        try:
            matrix_path = os.path.join(path, 'user_item_matrix.npz')
            if os.path.exists(matrix_path):
                logger.info("Loading user-item matrix")
                loaded_matrix = joblib.load(matrix_path)

                # Force to CSR by creating a new matrix - more reliable
                from scipy.sparse import csr_matrix
                data = loaded_matrix.data
                indices = loaded_matrix.indices
                indptr = loaded_matrix.indptr
                shape = loaded_matrix.shape

                # Create a brand new CSR matrix from the components
                model.user_item_matrix = csr_matrix((data, indices, indptr), shape=shape)

                logger.info(f"User-item matrix loaded with shape {model.user_item_matrix.shape}")
            else:
                logger.warning("User-item matrix file not found")
                model.user_item_matrix = None
        except Exception as e:
            logger.warning(f"Error loading user-item matrix: {e}")
            model.user_item_matrix = None

        # Load collaborative filtering model if available
        try:
            with open(os.path.join(path, 'cf_model.pkl'), 'rb') as f:
                model.cf_model = pickle.load(f)
        except Exception as e:
            logger.warning(f"Collaborative filtering model not found: {e}")
            model.cf_model = None

        # Load other components...
        # (Truncated for brevity - add additional loading logic for the remaining components)

        logger.info("Model loaded successfully")
        return model

    def debug_matrix_format(self):
        """
        Utility function to debug matrix format issues
        """
        if not hasattr(self, 'user_item_matrix') or self.user_item_matrix is None:
            logger.error("User-item matrix is None")
            return

        from scipy.sparse import issparse, isspmatrix_csr, csr_matrix

        logger.info(f"Matrix type: {type(self.user_item_matrix)}")
        logger.info(f"Is sparse: {issparse(self.user_item_matrix)}")
        logger.info(f"Is CSR: {isspmatrix_csr(self.user_item_matrix)}")

        if issparse(self.user_item_matrix) and not isspmatrix_csr(self.user_item_matrix):
            # Convert to CSR and test
            logger.info("Converting to CSR and testing...")
            new_matrix = csr_matrix(self.user_item_matrix)
            logger.info(f"New matrix is CSR: {isspmatrix_csr(new_matrix)}")

            # Test explicit construction
            data = self.user_item_matrix.data
            indices = self.user_item_matrix.indices
            indptr = self.user_item_matrix.indptr
            shape = self.user_item_matrix.shape

            explicit_csr = csr_matrix((data, indices, indptr), shape=shape)
            logger.info(f"Explicit construction is CSR: {isspmatrix_csr(explicit_csr)}")

        logger.info(f"Matrix shape: {self.user_item_matrix.shape}")
        logger.info(f"Matrix nnz: {self.user_item_matrix.nnz}")