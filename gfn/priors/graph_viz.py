import json
import graphviz
import os

def format_node_label(node_data):
    """Formats the label for a node based on its data."""
    if not node_data:
        return "Invalid Node" # Should not happen if called correctly

    role = node_data.get('role', 'Unknown')
    triples = node_data.get('triples', [])
    logical_rel = node_data.get('logical_rel', 'null')

    label_lines = []
    if role == 'C':
        label_lines.append("Condition:")
        if len(triples) > 1 and logical_rel != 'null':
            label_lines.append(f"(Logic: {logical_rel.upper()})")
        if not triples:
             label_lines.append("[Empty Condition]")
        else:
            label_lines.extend(triples) # Add each triple on a new line implicitly via graphviz
    elif role == 'D':
        label_lines.append("Decision:")
        if not triples:
             label_lines.append("[No Action/Outcome]")
        else:
             label_lines.extend(triples)
    else:
        label_lines.append(f"Role: {role}")
        label_lines.extend(triples)

    # Join lines with newline characters for Graphviz
    # Graphviz usually handles wrapping, but explicit newlines help structure
    # Escape special characters if necessary, though usually not needed for labels
    return "\n".join(label_lines)

def visualize_bfs_tree(bfs_tree, filename="decision_tree", view=False, format='png'):
    """
    Visualizes a decision tree stored in BFS list format using Graphviz.

    Args:
        bfs_tree (list): The tree structure as a list of nodes (dict or None).
        filename (str): The base name for the output file (without extension).
        view (bool): Whether to automatically open the generated file.
        format (str): The output format (e.g., 'png', 'svg', 'pdf').
    """
    if not bfs_tree or bfs_tree[0] is None:
        print(f"Skipping empty or invalid tree for {filename}")
        return

    # Changed from 'ortho' to 'polyline' to avoid issues with edge labels
    dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir': 'TB', 'splines': 'polyline'})
    dot.attr('node', shape='box', style='filled') # Default node style

    n = len(bfs_tree)
    for i, node_data in enumerate(bfs_tree):
        if node_data is None:
            continue # Skip null nodes

        # --- Define Node ---
        node_id = str(i)
        label = format_node_label(node_data)
        role = node_data.get('role')

        # Customize node appearance based on role
        if role == 'C':
            dot.node(node_id, label=label, shape='box', fillcolor='lightblue')
        elif role == 'D':
            dot.node(node_id, label=label, shape='ellipse', fillcolor='lightgrey')
        else: # Unknown role
             dot.node(node_id, label=label, fillcolor='red')


        # --- Define Edges (only for Conditional nodes) ---
        if role == 'C':
            # Left child (Condition False/Absent)
            left_child_idx = 2 * i + 1
            if left_child_idx < n and bfs_tree[left_child_idx] is not None:
                dot.edge(node_id, str(left_child_idx), label="Absent / False")

            # Right child (Condition True/Present)
            right_child_idx = 2 * i + 2
            if right_child_idx < n and bfs_tree[right_child_idx] is not None:
                dot.edge(node_id, str(right_child_idx), label="Present / True")

    # --- Render Graph ---
    try:
        # The render function saves the file and optionally opens it
        output_path = dot.render(filename, cleanup=True, format=format, view=view)
        print(f"Tree visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error rendering graph '{filename}': {e}")
        print("Make sure Graphviz is installed and in your system's PATH.")

def plot_tree(tree_path):
    if 'generated' in tree_path:
        with open(tree_path, "r") as f:
            bfs_tree = json.load(f)
    else:
        with open(tree_path, "r") as f:
            bfs_tree = [json.load(f)["bfs_tree"]]
    path_prefix = tree_path.split("/")[:-1]
    output_dir = "/".join(path_prefix) + "/decision_tree_viz"
    os.makedirs(output_dir, exist_ok=True)
    for idx, tree in enumerate(bfs_tree):
        file_base = os.path.join(output_dir, f"decision_tree_{idx}")
        # Set view=False to avoid xdg-open errors
        visualize_bfs_tree(tree, filename=file_base, view=False, format='png')

def main():
    tree_path = "0429_152340_processed_tree_NSCLC_0.json"
    plot_tree(tree_path)

if __name__ == "__main__":
    main()
