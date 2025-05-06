import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import argparse

def prepare_iris_dataset(output_dir: str, split_seed: int = 1, test_size: float = 0.2):
    """
    Loads the Iris dataset, splits it using the specified seed,
    and saves it as a CSV file with a 'Split' column.
    """
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Use standard feature names that are less likely to cause issues
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    target_name = 'target' # Standard name

    # Create DataFrame
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.DataFrame(y, columns=[target_name])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=test_size, random_state=split_seed, stratify=df_y
    )

    # Combine features and target for train and test sets
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Add 'Split' column
    df_train['Split'] = 'train'
    df_test['Split'] = 'test'

    # Combine train and test sets
    df_final = pd.concat([df_train, df_test], ignore_index=True)

    # Define output path
    file_name = f"iris_{split_seed}.csv"
    output_path = os.path.join(output_dir, file_name)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    df_final.to_csv(output_path, index=False)
    print(f"Successfully created '{output_path}'")
    print(f"Columns: {list(df_final.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Iris dataset for GFlowNet experiments.")
    parser.add_argument(
        "output_directory",
        help="Directory where the 'iris_1.csv' file will be saved (e.g., /path/to/datasets/iris or ./data/iris). This should be the 'iris' subdirectory."
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=1,
        help="Random seed for train/test split (default: 1)."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)."
    )

    args = parser.parse_args()

    # Ensure the output directory is the specific iris directory
    if os.path.basename(args.output_directory) != 'iris':
        print(f"Warning: Output directory '{args.output_directory}' does not end with 'iris'. Ensure this is the correct target directory.")

    prepare_iris_dataset(args.output_directory, args.split_seed, args.test_size) 