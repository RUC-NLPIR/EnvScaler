"""
step3-2 (optional): Select environments through deduplication and filtering.
1. Deduplicate by environment_summary
2. Filter by metrics thresholds
3. KMeans clustering deduplication
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans

from utils.process_file import read_file, save_file


def deduplicate_environments(env_list):
    """Deduplicate environments by environment_summary, keeping item with highest Modelability and Usefulness scores."""
    # key: environment_summary, value: best record
    best_env_map = {}

    for env in env_list:
        summary = env.get("environment_summary", "").strip()
        model_score = env.get("metrics", {}).get("modelability", 0)
        useful_score = env.get("metrics", {}).get("usefulness", 0)

        if summary not in best_env_map:
            best_env_map[summary] = env
        else:
            current_best = best_env_map[summary]
            best_model_score = current_best.get("metrics", {}).get("modelability", 0)
            best_useful_score = current_best.get("metrics", {}).get("usefulness", 0)

            # Prioritize Modelability
            if model_score > best_model_score:
                best_env_map[summary] = env
            elif model_score == best_model_score:
                # If Modelability is equal, compare Usefulness
                if useful_score > best_useful_score:
                    best_env_map[summary] = env

    return list(best_env_map.values())


def filter_environments(env_list, modelability_threshold=0, usefulness_threshold=0):
    """Filter environments by Modelability and Usefulness score thresholds."""
    filtered = []
    for env in env_list:
        model_score = env.get("metrics", {}).get("modelability", 0)
        useful_score = env.get("metrics", {}).get("usefulness", 0)

        if model_score >= modelability_threshold and useful_score >= usefulness_threshold:
            filtered.append(env)

    return filtered


def cluster_deduplicate(items: List[Dict[str, Any]], embedding_field, n_clusters: int) -> List[Dict[str, Any]]:
    """Cluster items by embeddings using KMeans and return the closest item to each cluster center."""
    if not items:
        return []

    # Extract all embeddings
    embeddings = np.array([np.array(item[embedding_field]) for item in items])

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    # Find closest item to center for each cluster
    closest_items = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_embeddings = embeddings[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]

        # Calculate distances and find minimum
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_index_in_cluster = cluster_indices[np.argmin(distances)]
        closest_items.append(items[closest_index_in_cluster])

    return closest_items


if __name__ == "__main__":
    # Configuration
    modelability_threshold = 7
    usefulness_threshold = 7
    n_clusters = 2
    # Read data
    data = read_file("stage1_collect_env_from_task/temp_result/step3_infered_env_description_with_embedding.json")
    print(len(data))
    # Deduplicate by environment_summary
    new_data = deduplicate_environments(data)
    print(len(new_data))
    # Filter by metrics
    new_data = filter_environments(new_data, modelability_threshold=modelability_threshold, usefulness_threshold=usefulness_threshold)
    new_data = cluster_deduplicate(new_data, embedding_field="env_summary_and_introduction_embedding", n_clusters=n_clusters)
    print(len(new_data))
    save_file("stage1_collect_env_from_task/temp_result/step3_infered_env_description_selected.json", new_data)
    # Save final result
    final_data = []
    for item in new_data:
        final_data.append({
            "task": item["task"],
            "environment_summary": item["environment_summary"],
            "environment_introduction": item["environment_introduction"],
        })
    print("Final result saved to: stage1_collect_env_from_task/final_result/env_description.json")
    save_file("stage1_collect_env_from_task/final_result/env_description.json", final_data)