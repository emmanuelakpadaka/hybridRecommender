"""
Evaluation metrics for recommendation systems.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import multiprocessing

from config import (
    DEFAULT_USER_ID_COL, DEFAULT_ITEM_ID_COL, DEFAULT_TIMESTAMP_COL,
    DEFAULT_N_JOBS
)

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_model(
        model: Any,
        events_df: pd.DataFrame,
        k: int = 10,
        n_users: int = 1000,
        user_id_col: str = DEFAULT_USER_ID_COL,
        item_id_col: str = DEFAULT_ITEM_ID_COL,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL
) -> Dict:
    """
    Evaluate the model using hit rate and mean reciprocal rank.

    Parameters:
    -----------
    model: Any
        The recommendation model to evaluate
    events_df: pd.DataFrame
        DataFrame containing user-item interactions for evaluation
    k: int
        Number of recommendations to generate
    n_users: int
        Number of users to evaluate
    user_id_col: str
        Column name for user IDs
    item_id_col: str
        Column name for item IDs
    timestamp_col: str
        Column name for timestamps

    Returns:
    --------
    dict: Evaluation metrics
    """
    logger.info(f"Evaluating model on up to {n_users} users with k={k}")

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(events_df[timestamp_col]):
        events_df[timestamp_col] = pd.to_datetime(events_df[timestamp_col], unit='ms')

    # Sort by timestamp
    events_df = events_df.sort_values(timestamp_col)

    # Get users with at least 2 interactions
    user_counts = events_df[user_id_col].value_counts()
    eligible_users = user_counts[user_counts >= 2].index.tolist()

    # Get the maximum user ID that's within bounds for the model
    max_user_idx = None
    if hasattr(model, 'cf_model') and model.cf_model is not None:
        max_user_idx = model.cf_model.user_factors.shape[0] - 1
        logger.info(f"Maximum user index for CF model: {max_user_idx}")

    # Limit number of test users
    np.random.seed(42)
    test_users = min(n_users, len(eligible_users))
    test_user_ids = np.random.choice(eligible_users, test_users, replace=False)

    # Filter out users that are known to be out of bounds
    if max_user_idx is not None:
        filtered_test_users = []
        for user_id in test_user_ids:
            if user_id not in model.user_mapper:
                continue
            user_idx = model.user_mapper[user_id]
            if user_idx <= max_user_idx:
                filtered_test_users.append(user_id)

        if len(filtered_test_users) < len(test_user_ids):
            logger.warning(
                f"Filtered out {len(test_user_ids) - len(filtered_test_users)} users that are out of bounds")

        test_user_ids = filtered_test_users
        logger.info(f"Evaluating on {len(test_user_ids)} users after filtering")

    # Simple (single-process) evaluation to avoid pickling issues
    total_hit_rate = 0
    total_mrr = 0
    valid_users = 0
    attempted_users = 0

    try:
        for user_id in test_user_ids:
            attempted_users += 1
            # Get user events
            user_events = events_df[events_df[user_id_col] == user_id]

            if len(user_events) < 2:
                continue

            # Hold out last event as test item
            test_event = user_events.iloc[-1]
            test_item = test_event[item_id_col]

            # Check if item is in our model
            if test_item not in model.item_mapper:
                continue

            # Get recommendations
            try:
                recommendations = model.recommend_for_user(user_id, n=k)
                if not recommendations:
                    continue

                rec_items = [item for item, _ in recommendations]

                # Calculate hit rate
                if test_item in rec_items:
                    total_hit_rate += 1
                    rank = rec_items.index(test_item) + 1
                    total_mrr += 1.0 / rank

                valid_users += 1

                # Log progress periodically
                if valid_users % 10 == 0:
                    logger.info(f"Evaluated {valid_users} users so far (attempted {attempted_users})")

            except Exception as e:
                logger.debug(f"Error evaluating user {user_id}: {e}")
                continue

        # Calculate final metrics
        if valid_users == 0:
            logger.warning("No valid users found for evaluation")
            return {'hit_rate@k': 0.0, 'mrr@k': 0.0, 'k': k, 'valid_users': 0, 'attempted_users': attempted_users}

        hit_rate = total_hit_rate / valid_users
        mrr = total_mrr / valid_users

        logger.info(
            f"Evaluation complete - Hit Rate@{k}: {hit_rate:.4f}, MRR@{k}: {mrr:.4f}, Valid Users: {valid_users}/{attempted_users}")

        return {
            'hit_rate@k': hit_rate,
            'mrr@k': mrr,
            'k': k,
            'valid_users': valid_users,
            'attempted_users': attempted_users
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'hit_rate@k': 0.0, 'mrr@k': 0.0, 'k': k, 'valid_users': 0, 'attempted_users': attempted_users}


def generate_recommendations(
        model: Any,
        user_id: Any,
        n: int = 10,
        include_explanation: bool = True
) -> str:
    """
    Generate and format recommendations for a user with explanations.

    Parameters:
    -----------
    model: Any
        Trained recommendation model
    user_id: Any
        User ID to recommend for
    n: int
        Number of recommendations
    include_explanation: bool
        Whether to include explanations

    Returns:
    --------
    str: Formatted recommendation results
    """
    # Get recommendations
    recommendations = model.recommend_for_user(user_id, n=n)

    if not recommendations:
        return f"No recommendations could be generated for user {user_id}."

    # Format results
    result = f"Top {n} recommendations for user {user_id}:\n\n"

    for i, (item_id, score) in enumerate(recommendations):
        # Get item category if available
        category_info = ""
        if hasattr(model, 'item_categories') and model.item_categories and item_id in model.item_categories:
            category = model.item_categories[item_id]
            category_info = f"Category: {category}\n"

        # Generate explanation if requested
        explanation = ""
        if include_explanation:
            # Get collaborative filtering similar items
            cf_similar = model.get_similar_items(item_id, n=2, method='cf')

            # Get content-based similar items
            content_similar = model.get_similar_items(item_id, n=2, method='content')

            explanation = "Recommended because:\n"

            if cf_similar:
                explanation += "- Similar to items that users like you have interacted with\n"

            if content_similar:
                explanation += "- Has similar features to items you've shown interest in\n"

            if category_info:
                explanation += "- Belongs to a category you've shown interest in\n"

        result += f"{i + 1}. Item: {item_id} (Score: {score:.4f})\n"
        result += f"{category_info}{explanation}\n"

    return result


def analyze_category_performance(
        model: Any,
        events_df: pd.DataFrame,
        top_n: int = 10,
        item_id_col: str = DEFAULT_ITEM_ID_COL,
        event_type_col: str = 'event'
) -> str:
    """
    Analyze the performance of different item categories.

    Parameters:
    -----------
    model: Any
        Trained recommendation model
    events_df: pd.DataFrame
        DataFrame containing user-item interactions
    top_n: int
        Number of top categories to show
    item_id_col: str
        Column name for item IDs
    event_type_col: str
        Column name for event types

    Returns:
    --------
    str: Category performance analysis
    """
    from collections import defaultdict

    if not hasattr(model, 'item_categories') or not model.item_categories:
        return "Category information not available."

    # Get event counts by category
    category_counts = defaultdict(int)
    category_views = defaultdict(int)
    category_carts = defaultdict(int)
    category_purchases = defaultdict(int)

    # Map events to categories
    for _, row in events_df.iterrows():
        item_id = row[item_id_col]
        event_type = row[event_type_col]

        if item_id in model.item_categories:
            category = model.item_categories[item_id]
            category_counts[category] += 1

            if event_type == 'view':
                category_views[category] += 1
            elif event_type == 'addtocart':
                category_carts[category] += 1
            elif event_type == 'transaction':
                category_purchases[category] += 1

    # Calculate conversion rates
    category_stats = {}
    for category, count in category_counts.items():
        views = category_views[category]
        carts = category_carts[category]
        purchases = category_purchases[category]

        # Avoid division by zero
        view_to_cart = carts / views if views > 0 else 0
        cart_to_purchase = purchases / carts if carts > 0 else 0
        view_to_purchase = purchases / views if views > 0 else 0

        category_stats[category] = {
            'total_interactions': count,
            'views': views,
            'carts': carts,
            'purchases': purchases,
            'view_to_cart_rate': view_to_cart,
            'cart_to_purchase_rate': cart_to_purchase,
            'view_to_purchase_rate': view_to_purchase
        }

    # Sort categories by total interactions
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: -x[1]['total_interactions']
    )[:top_n]

    # Format results
    result = f"Top {top_n} Category Performance Analysis:\n\n"

    for i, (category, stats) in enumerate(sorted_categories):
        result += f"{i + 1}. Category: {category}\n"
        result += f"   Total Interactions: {stats['total_interactions']}\n"
        result += f"   Views: {stats['views']}\n"
        result += f"   Add to Cart: {stats['carts']}\n"
        result += f"   Purchases: {stats['purchases']}\n"
        result += f"   View → Cart Rate: {stats['view_to_cart_rate']:.1%}\n"
        result += f"   Cart → Purchase Rate: {stats['cart_to_purchase_rate']:.1%}\n"
        result += f"   View → Purchase Rate: {stats['view_to_purchase_rate']:.1%}\n\n"

    return result