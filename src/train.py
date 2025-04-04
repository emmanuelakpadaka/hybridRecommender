"""
Main script for training the CPU-optimized hybrid recommendation system.
"""
import os
import logging
import argparse
import multiprocessing
from typing import Dict, List, Tuple, Optional

from config import (
    DEFAULT_OUTPUT_PATH, DEFAULT_FACTORS, DEFAULT_ITERATIONS, DEFAULT_REGULARIZATION,
    DEFAULT_N_JOBS, DEFAULT_REUSE_ARTIFACTS
)
from data.loaders import load_all_data
from models.hybrid_recommender import CPUHybridRecommender
from evaluation.metrics import evaluate_model, generate_recommendations, analyze_category_performance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_cpu_model(
        events_path: str,
        item_properties_path1: str,
        item_properties_path2: Optional[str] = None,
        category_tree_path: Optional[str] = None,
        output_path: str = DEFAULT_OUTPUT_PATH,
        factors: int = DEFAULT_FACTORS,
        iterations: int = DEFAULT_ITERATIONS,
        regularization: float = DEFAULT_REGULARIZATION,
        n_jobs: Optional[int] = None,
        reuse_artifacts: bool = DEFAULT_REUSE_ARTIFACTS
) -> CPUHybridRecommender:
    """
    Train a CPU-only hybrid recommendation model.
    Will reuse existing artifacts when available.

    Parameters:
    -----------
    events_path: str
        Path to events CSV file
    item_properties_path1: str
        Path to first item properties CSV file
    item_properties_path2: str, optional
        Path to second item properties CSV file
    category_tree_path: str, optional
        Path to category tree CSV file
    output_path: str
        Path to save model and artifacts
    factors: int
        Number of latent factors for ALS
    iterations: int
        Number of training iterations
    regularization: float
        Regularization parameter for ALS
    n_jobs: int, optional
        Number of parallel jobs (defaults to CPU count - 1)
    reuse_artifacts: bool
        Whether to reuse existing artifacts when available

    Returns:
    --------
    CPUHybridRecommender: Trained model
    """
    # Set number of jobs based on available CPUs
    if n_jobs is None:
        n_jobs = DEFAULT_N_JOBS

    logger.info(f"Starting CPU-only model training with {n_jobs} parallel processes")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load data
    logger.info("Loading data")
    events_df, item_properties_df, category_tree_df = load_all_data(
        events_path,
        item_properties_path1,
        item_properties_path2,
        category_tree_path
    )

    # Initialize model
    logger.info("Initializing CPU-only recommendation model")
    model = CPUHybridRecommender(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        output_path=output_path,
        n_jobs=n_jobs
    )

    # Train hybrid model with artifact reuse
    logger.info("Training hybrid model with artifact reuse")
    model.train_hybrid_model(events_df, item_properties_df, category_tree_df, reuse_artifacts=reuse_artifacts)

    # Evaluate model
    logger.info("Evaluating model")
    evaluation = evaluate_model(model, events_df)

    logger.info(f"Model evaluation: {evaluation}")
    logger.info(f"Model saved to {output_path}")

    return model


def simple_recommender(
        events_path: str,
        output_path: str = DEFAULT_OUTPUT_PATH,
        reuse_artifacts: bool = DEFAULT_REUSE_ARTIFACTS
) -> CPUHybridRecommender:
    """
    Basic recommender pipeline for small datasets or quick testing.
    Only trains a simpler collaborative filtering model with minimal preprocessing.
    Will reuse existing artifacts when available.

    Parameters:
    -----------
    events_path: str
        Path to events CSV file
    output_path: str
        Path to save model
    reuse_artifacts: bool
        Whether to reuse existing artifacts when available

    Returns:
    --------
    CPUHybridRecommender: Trained simple model
    """
    logger.info("Running simple recommendation pipeline")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load events data
    events_df, _, _ = load_all_data(events_path, None, None, None)

    # Initialize model with minimal parameters
    model = CPUHybridRecommender(
        factors=50,  # Fewer factors for faster training
        iterations=5,  # Fewer iterations for faster training
        output_path=output_path,
        n_jobs=2  # Use minimal parallelization
    )

    # Train only collaborative filtering model
    model.preprocess_data(events_df, reuse_artifacts=reuse_artifacts)
    model.train_collaborative_filtering()
    model.save_model()

    logger.info("Simple model training complete")
    return model


def main():
    """
    Main function to train and evaluate the recommendation system.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate a CPU-optimized recommendation system')
    parser.add_argument('--events', type=str, required=True, help='Path to events CSV file')
    parser.add_argument('--properties1', type=str, required=False, help='Path to first item properties CSV file')
    parser.add_argument('--properties2', type=str, required=False, help='Path to second item properties CSV file')
    parser.add_argument('--categories', type=str, required=False, help='Path to category tree CSV file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH, help='Path to save model and artifacts')
    parser.add_argument('--factors', type=int, default=DEFAULT_FACTORS, help='Number of latent factors for ALS')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, help='Number of iterations for training')
    parser.add_argument('--regularization', type=float, default=DEFAULT_REGULARIZATION, help='Regularization for ALS')
    parser.add_argument('--simple', action='store_true', help='Use simple training mode for faster results')
    parser.add_argument('--jobs', type=int, default=None, help='Number of parallel jobs')
    parser.add_argument('--no-reuse', action='store_true', help='Disable reuse of existing artifacts')

    args = parser.parse_args()

    # Determine if we should run in simple mode
    if args.simple:
        model = simple_recommender(
            events_path=args.events,
            output_path=args.output,
            reuse_artifacts=not args.no_reuse
        )
    else:
        model = train_cpu_model(
            events_path=args.events,
            item_properties_path1=args.properties1,
            item_properties_path2=args.properties2,
            category_tree_path=args.categories,
            output_path=args.output,
            factors=args.factors,
            iterations=args.iterations,
            regularization=args.regularization,
            n_jobs=args.jobs,
            reuse_artifacts=not args.no_reuse
        )

    # Generate sample recommendations
    if model is not None and len(model.user_mapper) > 0:
        # Find a sample user
        sample_user = list(model.user_mapper.keys())[0]
        logger.info(f"Generating sample recommendations for user {sample_user}")
        recommendations = generate_recommendations(model, sample_user, n=5)
        logger.info(recommendations)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()