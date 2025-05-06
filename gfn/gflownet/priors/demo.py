import numpy as np
from prior_generation import calculate_average_similarity
# Assuming you have generated structural_priors.json previously
# e.g., using a function like generate_structural_priors(...)

# 1. Define your new tree in the numerical format (list of lists/numpy array)
#    Make sure its features/classes correspond to the mappings in priors_json!
#    (You might get mappings by calling json_tree.process_json_trees(priors_json_path))
new_tree_numerical = [[0, 3, 0.3, -1, 0, np.nan, np.nan, np.nan],
                      [1, -1, -1, -1, 0, 0.1, 0.2, 0.7],
                      [0, 3, 0.7, -1, 0, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan, np.nan],
                      [1, -1, -1, -1, 0, 0.2, 0.3, 0.5],
                      [1, -1, -1, -1, 0, 0.4, 0.5, 0.1],
                      [np.nan, np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan, np.nan],
                      [0, np.nan, np.nan, np.nan, np.nan]]

# 2. Specify the path to your prior trees file
priors_file = "/data/hzy/xh/dtfl/project/data/0503_081537/WD/structural_priors.json"

# 3. (Optional) Define custom bounds if needed, otherwise use the default bounds (0, 1) for all features
custom_bounds = [(0, 1), (0, 10), (-5, 5), (0, 1), (0, 1)] # Example

# 4. Calculate the average similarity
# Using default bounds (0, 1 for all features)
avg_sim_default_bounds = calculate_average_similarity(
    new_tree_numerical=new_tree_numerical,
    priors_json=priors_file,
    comp_dist=True, # Or False
    dist_weight=0.5
)
print(f"Average similarity (default bounds): {avg_sim_default_bounds:.4f}")

# Using custom bounds
avg_sim_custom_bounds = calculate_average_similarity(
    new_tree_numerical=new_tree_numerical,
    priors_json=priors_file,
    comp_dist=True,
    dist_weight=0.5,
    bounds=custom_bounds
)
print(f"Average similarity (custom bounds): {avg_sim_custom_bounds:.4f}")
