import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Feature names - same as in raisin dataset
FEATURE_NAMES = [
    "Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", 
    "ConvexArea", "Extent", "Perimeter"
]

# Feature ranges based on raisin dataset
FEATURE_RANGES = {
    "Area": (30000, 210000),
    "MajorAxisLength": (250, 700),
    "MinorAxisLength": (150, 450),
    "Eccentricity": (0.35, 0.95),
    "ConvexArea": (33000, 220000),
    "Extent": (0.45, 0.85),
    "Perimeter": (700, 2300)
}

def create_decision_tree(depth=5):
    """Create a binary decision tree of specified depth."""
    # Node structure: [node_type, feature_index, threshold, parent_index, depth, class0_prob, class1_prob]
    # node_type: 0 for internal nodes, 1 for leaf nodes
    # For internal nodes, class probabilities are null
    # For leaf nodes, feature_index and threshold are -1
    
    # Initialize tree with the root node
    tree = []
    max_nodes = 2**(depth+1) - 1
    
    # Define exact thresholds for each feature for a more deterministic tree
    thresholds = [
        [100000],  # Area (0)
        [400],     # MajorAxisLength (1)
        [250],     # MinorAxisLength (2)
        [0.70],    # Eccentricity (3)
        [105000],  # ConvexArea (4)
        [0.65],    # Extent (5)
        [1200]     # Perimeter (6)
    ]
    
    # Build a complete binary tree with specific features at each level
    for i in range(max_nodes):
        if i >= 2**(depth) - 1:  # Leaf nodes
            # Leaf node with pure class (alternating 0 and 1)
            if i % 2 == 0:
                tree.append([1, -1, -1.0, -1, 0, 1.0, 0.0])  # Class 0
            else:
                tree.append([1, -1, -1.0, -1, 0, 0.0, 1.0])  # Class 1
        else:  # Internal nodes
            level = int(np.floor(np.log2(i+1)))
            feature_idx = level % len(FEATURE_NAMES)  # Cycle through features
            threshold_value = thresholds[feature_idx][0]
            
            if feature_idx in [3, 5]:  # Eccentricity, Extent
                tree.append([0, feature_idx, threshold_value, -1, 0, None, None])
            else:
                tree.append([0, feature_idx, float(threshold_value), -1, 0, None, None])
    
    # Fill remaining nodes with null to maintain proper BFS structure
    while len(tree) < max_nodes:
        tree.append(None)
        
    return tree

def print_tree(tree):
    """Print a human-readable version of the tree."""
    print("Decision Tree Structure:")
    for i, node in enumerate(tree):
        if node is None:
            continue
        if node[0] == 0:  # Internal node
            feature_name = FEATURE_NAMES[node[1]]
            threshold = node[2]
            print(f"Node {i}: Split on {feature_name} <= {threshold}")
        else:  # Leaf node
            class_pred = 0 if node[5] > node[6] else 1
            print(f"Node {i}: Leaf (Class {class_pred})")

def traverse_tree(sample, tree):
    """Traverse the tree to classify a sample."""
    node_idx = 0
    path = [node_idx]
    
    while node_idx < len(tree) and tree[node_idx] is not None and tree[node_idx][0] == 0:  # While not leaf node
        feature_idx = tree[node_idx][1]
        threshold = tree[node_idx][2]
        
        if sample[feature_idx] <= threshold:
            node_idx = 2 * node_idx + 1  # Go to left child
        else:
            node_idx = 2 * node_idx + 2  # Go to right child
        
        path.append(node_idx)
            
        if node_idx >= len(tree) or tree[node_idx] is None:
            # Return default class if we hit a null node
            return 0
    
    # Return class based on leaf node probabilities
    if tree[node_idx][5] > tree[node_idx][6]:
        return 0
    else:
        return 1

def explain_classification(sample, tree):
    """Explain how a sample is classified by the tree."""
    node_idx = 0
    path = []
    
    while node_idx < len(tree) and tree[node_idx] is not None and tree[node_idx][0] == 0:
        feature_idx = tree[node_idx][1]
        threshold = tree[node_idx][2]
        feature_name = FEATURE_NAMES[feature_idx]
        feature_value = sample[feature_idx]
        
        decision = "≤" if feature_value <= threshold else ">"
        path.append(f"Node {node_idx}: {feature_name}={feature_value:.2f} {decision} {threshold:.2f}")
        
        if feature_value <= threshold:
            node_idx = 2 * node_idx + 1
        else:
            node_idx = 2 * node_idx + 2
            
        if node_idx >= len(tree) or tree[node_idx] is None:
            return path + ["Invalid path - reached null node"]
    
    cls = 0 if tree[node_idx][5] > tree[node_idx][6] else 1
    path.append(f"Node {node_idx}: Leaf node - Class {cls}")
    return path

def generate_sample_for_class(tree, target_class):
    """Generate a sample that will be classified as the target class."""
    max_attempts = 1000
    for _ in range(max_attempts):
        # Generate random sample
        sample = [
            np.random.uniform(*FEATURE_RANGES["Area"]),
            np.random.uniform(*FEATURE_RANGES["MajorAxisLength"]),
            np.random.uniform(*FEATURE_RANGES["MinorAxisLength"]),
            np.random.uniform(*FEATURE_RANGES["Eccentricity"]),
            np.random.uniform(*FEATURE_RANGES["ConvexArea"]),
            np.random.uniform(*FEATURE_RANGES["Extent"]),
            np.random.uniform(*FEATURE_RANGES["Perimeter"])
        ]
        
        # Check if it gets classified as the target class
        cls = traverse_tree(sample, tree)
        if cls == target_class:
            # Format the sample correctly
            formatted_sample = []
            for i, val in enumerate(sample):
                if i in [3, 5]:  # Eccentricity, Extent
                    formatted_sample.append(round(val, 2))
                else:  # All other features
                    formatted_sample.append(int(val) if i in [0, 4] else round(val, 2))
            
            return formatted_sample, target_class
    
    raise ValueError(f"Failed to generate sample for class {target_class} after {max_attempts} attempts")

def generate_dataset(tree, n_train=10, n_test=50):
    """Generate a dataset with n_train training samples and n_test test samples."""
    data = []
    
    # Generate balanced training samples
    for _ in range(n_train // 2):
        sample, _ = generate_sample_for_class(tree, 0)
        data.append(sample + [0, "train"])
        
        sample, _ = generate_sample_for_class(tree, 1)
        data.append(sample + [1, "train"])
    
    # Generate balanced test samples
    for _ in range(n_test // 2):
        sample, _ = generate_sample_for_class(tree, 0)
        data.append(sample + [0, "test"])
        
        sample, _ = generate_sample_for_class(tree, 1)
        data.append(sample + [1, "test"])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=FEATURE_NAMES + ["class", "Split"])
    
    return df

def main():
    # Create output directory if it doesn't exist
    Path("dt-gfn/data/syn_data").mkdir(parents=True, exist_ok=True)
    
    # Create a binary decision tree of depth 5
    tree = create_decision_tree(depth=5)
    
    # Print the tree structure
    print_tree(tree)
    
    # Generate dataset
    df = generate_dataset(tree, n_train=10, n_test=50)
    
    # Save tree to JSON file (matching raisin_prior_trees_for_compare.json format)
    with open("dt-gfn/data/syn_data/syn_prior_trees_for_compare.json", "w") as f:
        json.dump([tree], f, indent=2)
    
    # Save dataset to CSV file (matching raisin_1.csv format)
    df.to_csv("dt-gfn/data/syn_data/syn_1.csv", index=False)
    
    print(f"\nTree saved to dt-gfn/data/syn_data/syn_prior_trees_for_compare.json")
    print(f"Dataset saved to dt-gfn/data/syn_data/syn_1.csv")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df['class'].value_counts().to_dict()}")
    print(f"Split distribution: {df['Split'].value_counts().to_dict()}")
    
    # Verify the dataset is perfectly classified by the tree
    correct = 0
    incorrect = []
    
    for idx, row in df.iterrows():
        sample = [
            row["Area"],
            row["MajorAxisLength"],
            row["MinorAxisLength"],
            row["Eccentricity"],
            row["ConvexArea"],
            row["Extent"],
            row["Perimeter"]
        ]
        predicted_class = traverse_tree(sample, tree)
        if predicted_class == row["class"]:
            correct += 1
        else:
            incorrect.append((idx, row, predicted_class))
    
    accuracy = correct / len(df)
    print(f"\nClassification accuracy: {accuracy:.2%}")
    
    if incorrect:
        print("\nIncorrectly classified samples:")
        for idx, row, pred in incorrect:
            print(f"Sample {idx}:")
            print(f"  Features: {dict(zip(FEATURE_NAMES, [row[f] for f in FEATURE_NAMES]))}")
            print(f"  True class: {int(row['class'])}")
            print(f"  Predicted class: {pred}")
            print("  Classification path:")
            for step in explain_classification([row[f] for f in FEATURE_NAMES], tree):
                print(f"    {step}")
            print()
    
    assert accuracy == 1.0, "The tree does not perfectly classify the dataset!"
    print("\nAll samples are correctly classified by the decision tree.")

if __name__ == "__main__":
    main() 