"""
Helper functions for the recommendation system, particularly for multiprocessing tasks.
"""
import logging
import pandas as pd
import numpy as np
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)

# Define helper functions for multiprocessing compatibility
def process_user_chunk_helper(args):
    """
    Helper function to process user sequences for multiprocessing.

    Parameters:
    -----------
    args: tuple
        (user_ids, sorted_events, item_id_col, user_id_col, item_mapper, user_mapper)

    Returns:
    --------
    dict: User sequences
    """
    user_ids, sorted_events, item_id_col, user_id_col, item_mapper, user_mapper = args

    chunk_sequences = {}
    for user_id in user_ids:
        user_data = sorted_events[sorted_events[user_id_col] == user_id]
        # Create sequence of item indices
        sequence = []
        for item in user_data[item_id_col].values:
            idx = item_mapper.get(item, -1)
            if idx >= 0:
                sequence.append(idx)

        if sequence:
            user_idx = user_mapper.get(user_id, -1)
            if user_idx >= 0:
                chunk_sequences[user_idx] = sequence

    return chunk_sequences


def process_property_chunk_helper(args):
    """
    Helper function to process property chunks for multiprocessing.

    Parameters:
    -----------
    args: tuple
        (chunk_df, timestamp_col)

    Returns:
    --------
    DataFrame: Processed chunk
    """
    chunk_df, timestamp_col = args
    return (chunk_df.sort_values(timestamp_col)
            .groupby(['itemid', 'property'])
            .last()
            .reset_index())


def compute_similarities_helper(args):
    """
    Helper function to compute category similarities for multiprocessing.

    Parameters:
    -----------
    args: tuple
        (cat_chunk, categories, cat_mapping, G, ancestors_cache)

    Returns:
    --------
    list: List of (i, j, sim_value) tuples
    """
    cat_chunk, categories, cat_mapping, G, ancestors_cache = args

    chunk_results = []
    for cat1 in cat_chunk:
        i = cat_mapping[cat1]
        for cat2 in categories:
            j = cat_mapping[cat2]

            if cat1 == cat2:
                sim_value = 1.0  # Same category
            else:
                try:
                    # Shortest path distance (inverted and normalized)
                    distance = nx.shortest_path_length(G, cat1, cat2)
                    sim_value = 1.0 / (1.0 + distance)
                except nx.NetworkXNoPath:
                    try:
                        # Try reverse direction
                        distance = nx.shortest_path_length(G, cat2, cat1)
                        sim_value = 1.0 / (1.0 + distance)
                    except nx.NetworkXNoPath:
                        # Find closest common ancestor
                        ancestors1 = ancestors_cache[cat1]
                        ancestors2 = ancestors_cache[cat2]
                        common = ancestors1.intersection(ancestors2)

                        if common:
                            # Use minimum distance to common ancestor
                            min_dist = float('inf')
                            for ancestor in common:
                                try:
                                    d1 = nx.shortest_path_length(G, cat1, ancestor)
                                    d2 = nx.shortest_path_length(G, cat2, ancestor)
                                    min_dist = min(min_dist, d1 + d2)
                                except:
                                    continue

                            if min_dist < float('inf'):
                                sim_value = 1.0 / (1.0 + min_dist)
                            else:
                                sim_value = 0.0
                        else:
                            sim_value = 0.0

            chunk_results.append((i, j, sim_value))

    return chunk_results


def evaluate_user_chunk_helper(args):
    """
    Helper function to evaluate user chunks for multiprocessing.

    Parameters:
    -----------
    args: tuple
        (user_ids, events_df, model, k, item_id_col, user_id_col, timestamp_col)

    Returns:
    --------
    tuple: (hit_rate, mrr, valid_users)
    """
    user_ids, events_df, model, k, item_id_col, user_id_col, timestamp_col = args

    hit_rate = 0
    mrr = 0
    valid_users = 0

    for user_id in user_ids:
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
            rec_items = [item for item, _ in recommendations]

            # Calculate hit rate
            if test_item in rec_items:
                hit_rate += 1
                rank = rec_items.index(test_item) + 1
                mrr += 1.0 / rank

            valid_users += 1
        except Exception as e:
            logger.error(f"Error evaluating user {user_id}: {e}")
            continue

    return hit_rate, mrr, valid_users


def properties_to_text(item_props):
    """
    Convert properties to text representation

    Parameters:
    -----------
    item_props: DataFrame
        DataFrame containing properties for an item

    Returns:
    --------
    str: Text representation
    """
    return " ".join(f"{prop}_{val}" for prop, val in
                    zip(item_props['property'], item_props['value'])
                    if pd.notna(val))