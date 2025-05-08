import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, classification_report
from collections import deque
import os

# Ensure the script can find the CSV relative to its location
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "wine.csv")
output_dir = script_dir # Save outputs in the same directory

# --- BFS Conversion Function (Adapted for Wine) ---
def convert_sklearn_tree_to_bfs_numerical_wine(dt_classifier, feature_names_ordered, global_wine_classes_str_list):
    """
    Converts a trained scikit-learn DecisionTreeClassifier for Wine
    into a BFS numerical list format.
    """
    sklearn_tree = dt_classifier.tree_
    num_classes_global = len(global_wine_classes_str_list)

    max_depth = sklearn_tree.max_depth
    if sklearn_tree.node_count == 0:
        return []
    if max_depth <= 0:
        bfs_list_size = 1
    else:
        bfs_list_size = 2**(max_depth + 1) - 1

    bfs_nodes = [None] * bfs_list_size
    queue = deque()
    queue.append((0, 0))  # (sklearn_node_id, bfs_list_index)

    while queue:
        current_sklearn_id, current_bfs_idx = queue.popleft()

        if current_sklearn_id == _tree.TREE_LEAF:
            continue
        if current_bfs_idx >= bfs_list_size:
            print(f"Warning: BFS index {current_bfs_idx} exceeds list size {bfs_list_size}. Skipping node.")
            continue

        node_values_from_sklearn = sklearn_tree.value[current_sklearn_id][0]
        total_samples_in_node = np.sum(node_values_from_sklearn)

        node_probs_standard_length = [0.0] * num_classes_global

        if total_samples_in_node > 0:
            temp_probs_sklearn_order = (node_values_from_sklearn / total_samples_in_node)
            for i, class_val_in_tree in enumerate(dt_classifier.classes_):
                try:
                    # Convert class value (likely int like 3, 4, ...) to string for lookup
                    global_idx = global_wine_classes_str_list.index(str(int(class_val_in_tree)))
                    node_probs_standard_length[global_idx] = float(temp_probs_sklearn_order[i])
                except ValueError:
                    print(f"Warning: Tree class {class_val_in_tree} not found in global list {global_wine_classes_str_list}.")
        else:
            node_probs_standard_length = [1.0 / num_classes_global] * num_classes_global

        node_probs_list = node_probs_standard_length

        if sklearn_tree.children_left[current_sklearn_id] == _tree.TREE_LEAF:
            bfs_node_representation = [1, -1, -1.0, -1, 0] + [float(p) for p in node_probs_list]
        else:
            feature_idx_sklearn = int(sklearn_tree.feature[current_sklearn_id])
            threshold = float(sklearn_tree.threshold[current_sklearn_id])
            original_feature_idx = feature_idx_sklearn # Assuming direct mapping
            # Represent internal node probs as nan for JSON compatibility
            bfs_node_representation = [0, original_feature_idx, threshold, -1, 0] + [np.nan] * num_classes_global

            left_child_sklearn_id = sklearn_tree.children_left[current_sklearn_id]
            right_child_sklearn_id = sklearn_tree.children_right[current_sklearn_id]
            left_bfs_idx = 2 * current_bfs_idx + 1
            right_bfs_idx = 2 * current_bfs_idx + 2

            if left_child_sklearn_id != _tree.TREE_LEAF and left_bfs_idx < bfs_list_size:
                queue.append((left_child_sklearn_id, left_bfs_idx))
            if right_child_sklearn_id != _tree.TREE_LEAF and right_bfs_idx < bfs_list_size:
                queue.append((right_child_sklearn_id, right_bfs_idx))

        bfs_nodes[current_bfs_idx] = bfs_node_representation

    # Convert numpy types and nan to JSON serializable types
    final_bfs_nodes = []
    for node_data in bfs_nodes:
        if node_data is not None:
            processed_node = []
            for val in node_data:
                if isinstance(val, (np.integer, np.longlong)):
                    processed_node.append(int(val))
                elif isinstance(val, (np.floating, np.longdouble)):
                    # Handle potential inf/-inf values before converting to float
                    if np.isinf(val):
                         processed_node.append(str(val)) # Store inf as string 'inf' or '-inf'
                    elif np.isnan(val):
                         processed_node.append(None)
                    else:
                         processed_node.append(float(val))
                elif isinstance(val, float) and np.isnan(val):
                    processed_node.append(None) # nan -> None
                else:
                    processed_node.append(val)
            final_bfs_nodes.append(processed_node)
        else:
            final_bfs_nodes.append(None)

    return final_bfs_nodes

# --- Main Script Logic ---
try:
    wine_full_df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit()

# Identify feature columns (first 11 columns) and target/split columns
feature_cols = wine_full_df.columns[:11].tolist()
target_col = 'class'
split_col = 'Split'

# Check if Split column exists
if split_col not in wine_full_df.columns:
    print(f"Error: '{split_col}' column not found in {csv_path}.")
    exit()

# Determine unique classes and create global list
all_classes = sorted(wine_full_df[target_col].unique())
GLOBAL_WINE_CLASSES_STR = [str(int(c)) for c in all_classes]
print(f"Detected Wine classes: {GLOBAL_WINE_CLASSES_STR}")

# Separate fixed test set
fixed_test_df = wine_full_df[wine_full_df[split_col] == 'test'].copy()
if fixed_test_df.empty:
     print("Warning: No samples found with Split='test'. Cannot evaluate priors.")
     X_fixed_test = pd.DataFrame(columns=feature_cols)
     y_fixed_test = pd.Series(dtype=wine_full_df[target_col].dtype)
else:
    X_fixed_test = fixed_test_df[feature_cols]
    y_fixed_test = fixed_test_df[target_col]
print(f"Loaded fixed test set with {len(fixed_test_df)} samples.")

# Separate initial training set
initial_train_df = wine_full_df[wine_full_df[split_col] == 'train'].copy()
print(f"Loaded initial training set with {len(initial_train_df)} samples.")

if initial_train_df.empty:
    print("Error: No training samples found (Split='train'). Cannot proceed.")
    exit()

# Randomly sample 20 for GFlowNet training (ensure reproducibility)
n_gfn_train = 20
if len(initial_train_df) < n_gfn_train:
    print(f"Warning: Requested {n_gfn_train} GFN training samples, but only {len(initial_train_df)} available. Using all.")
    n_gfn_train = len(initial_train_df)

gflownet_train_df = initial_train_df.sample(n=n_gfn_train, random_state=42)
print(f"Sampled {len(gflownet_train_df)} samples for GFlowNet training.")

# Combine GFN training and fixed test for wine_0.csv
output_data_for_wine_0 = pd.concat([gflownet_train_df, fixed_test_df])
output_file_0 = os.path.join(output_dir, "wine_0.csv")
output_data_for_wine_0.to_csv(output_file_0, index=False)
print(f"Saved {output_file_0} with {len(gflownet_train_df)} GFN train and {len(fixed_test_df)} test samples.")

# Define the pool for prior generation (initial train excluding GFN train)
prior_pool_df = initial_train_df.drop(gflownet_train_df.index)
print(f"Pool for prior generation has {len(prior_pool_df)} samples.")

# Generate 8 prior trees
num_prior_trees = 8
prior_sample_size = 100
all_prior_bfs_trees = []
prior_trees_evaluation_metrics = []

if len(prior_pool_df) == 0:
    print("Error: Prior pool is empty after taking GFN samples. Cannot generate prior trees.")
    exit()
elif len(prior_pool_df) < prior_sample_size:
     print(f"Warning: Prior pool size ({len(prior_pool_df)}) is less than sample size ({prior_sample_size}). Sampling with replacement from the smaller pool.")

for i in range(num_prior_trees):
    print(f"--- Generating Prior Tree {i+1}/{num_prior_trees} ---")
    # Sample 100 with replacement from the prior pool
    current_sample_df = prior_pool_df.sample(n=prior_sample_size, replace=True, random_state=42 + i)

    X_prior_train = current_sample_df[feature_cols]
    y_prior_train = current_sample_df[target_col]

    # Ensure target variable is categorical with all possible classes
    y_prior_train_cat = pd.Categorical(y_prior_train, categories=all_classes)

    # Train Decision Tree (max_depth=4)
    clf_prior = DecisionTreeClassifier(max_depth=4, random_state=100 + i)
    clf_prior.fit(X_prior_train, y_prior_train_cat)

    # Evaluate on the fixed test set (if it exists)
    print(f"Evaluating Prior Tree {i+1} on fixed test set...")
    if not X_fixed_test.empty:
        y_pred_on_test = clf_prior.predict(X_fixed_test)
        accuracy_on_test = accuracy_score(y_fixed_test, y_pred_on_test)
        try:
            # Use determined GLOBAL_WINE_CLASSES_STR for target names
            report_dict_on_test = classification_report(y_fixed_test, y_pred_on_test, zero_division=0, target_names=GLOBAL_WINE_CLASSES_STR, output_dict=True)
            report_str_on_test = classification_report(y_fixed_test, y_pred_on_test, zero_division=0, target_names=GLOBAL_WINE_CLASSES_STR)
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            report_dict_on_test = {"error": str(e)}
            report_str_on_test = f"Error generating report: {e}"
        print(f"Fixed test set accuracy: {accuracy_on_test:.4f}")
        print("Fixed test set classification report:")
        print(report_str_on_test)
    else:
        print("Skipping evaluation as test set is empty.")
        accuracy_on_test = np.nan
        report_dict_on_test = {}
        report_str_on_test = "No test set available for evaluation."

    # Store metrics
    fold_metrics = {
        "iteration": i + 1,
        "accuracy": accuracy_on_test,
        "classification_report": report_dict_on_test
    }
    prior_trees_evaluation_metrics.append(fold_metrics)

    # Convert to BFS numerical format
    bfs_tree = convert_sklearn_tree_to_bfs_numerical_wine(clf_prior, feature_cols, GLOBAL_WINE_CLASSES_STR)
    all_prior_bfs_trees.append(bfs_tree)

    # Save individual tree
    individual_tree_filename = os.path.join(output_dir, f"wine_prior_tree_iter_{i+1}.json")
    try:
        with open(individual_tree_filename, 'w') as f:
            json.dump([bfs_tree], f, indent=2)
        print(f"Saved individual prior BFS tree to {individual_tree_filename}")
    except Exception as e:
        print(f"Failed to save individual prior tree {individual_tree_filename}: {e}")
    print(f"--- Prior Tree {i+1} Generation Complete ---\n")


# Save combined prior trees
combined_priors_filename = os.path.join(output_dir, "wine_prior_trees_for_compare.json")
try:
    with open(combined_priors_filename, 'w') as f:
        json.dump(all_prior_bfs_trees, f, indent=2)
    print(f"Successfully saved {len(all_prior_bfs_trees)} prior BFS trees to {combined_priors_filename}")
except Exception as e:
    print(f"Failed to save combined prior trees file: {e}")

# Save evaluation metrics
evaluation_metrics_filename = os.path.join(output_dir, "wine_prior_trees_evaluation_metrics.json")
try:
    with open(evaluation_metrics_filename, 'w') as f:
        json.dump(prior_trees_evaluation_metrics, f, indent=2)
    print(f"Successfully saved evaluation metrics to {evaluation_metrics_filename}")
except Exception as e:
    print(f"Failed to save evaluation metrics file: {e}")

print("\nWine prior generation and evaluation complete.")

# Calculate bounds from the full dataset
print("\nCalculating feature bounds from full dataset...")
bounds = []
try:
    for col in feature_cols:
        min_val = wine_full_df[col].min()
        max_val = wine_full_df[col].max()
        bounds.append((float(min_val), float(max_val)))
    print("Bounds calculated.")
    bounds_filename = os.path.join(output_dir, "wine_feature_bounds.json")
    with open(bounds_filename, 'w') as f:
        json.dump(dict(zip(feature_cols, bounds)), f, indent=2)
    print(f"Saved feature bounds to {bounds_filename}")

except Exception as e:
    print(f"Failed to calculate or save bounds: {e}")
    print("Bounds calculation failed.") 