import json
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
from gfn_trees import compare_trees
from json_tree import process_json_trees

def predict_with_tree(tree, sample, debug=False):
    """
    使用决策树对单个样本进行预测，增加调试输出
    Logic: Left = False / Absent, Right = True / Present
    
    Args:
        tree: BFS格式的树
        sample: 单个样本的特征字典
        debug: 是否启用调试模式
    
    Returns:
        预测的类别
    """
    current_node_idx = 0
    
    if debug:
        print("开始预测样本:", sample)
    
    path = []
    while True:
        # Boundary check for current index
        if current_node_idx >= len(tree) or tree[current_node_idx] is None:
            if debug:
                print(f"Reached invalid node index {current_node_idx} or None node. Returning empty class.")
                print(f"Path taken: {path}")
            return tuple([]) # Return empty class if we fall off the tree
            
        current_node = tree[current_node_idx]
        path.append(current_node_idx)
        
        if debug:
            print(f"当前节点 {current_node_idx}:", current_node)
        
        # 如果是叶节点，返回其类别
        if current_node["role"] == "D":
            result = tuple(sorted(current_node["triples"] if current_node else []))
            if debug:
                print(f"到达叶节点，返回类别: {result}")
                print(f"完整路径: {path}")
            return result
        
        # 如果是条件节点
        if current_node["role"] == "C":
            conditions = current_node["triples"]
            # After unfolding, expect exactly one condition
            if not conditions or len(conditions) != 1:
                 print(f"Warning: Expected 1 condition at node {current_node_idx}, found {len(conditions)}. Path: {path}. Returning empty class.")
                 return tuple([]) # Treat as error/undefined path
            
            condition_tested = conditions[0]
            
            if debug:
                print(f"条件节点条件: {condition_tested}")
                value_in_sample = sample.get(condition_tested, 0) # Default to 0 if feature not present
                print(f"样本中特征 '{condition_tested}' 的值: {value_in_sample}")
            
            # Check if condition is satisfied (present/True -> value is 1)
            condition_satisfied = (condition_tested in sample and sample[condition_tested] == 1)
            
            # 根据条件决定走左子树 (False) 还是右子树 (True)
            if condition_satisfied:
                # Go Right (True)
                next_node_idx = 2 * current_node_idx + 2
                if debug:
                    print(f"条件 '{condition_tested}' 满足 (1), 走右子树 -> {next_node_idx}")
            else:
                # Go Left (False)
                next_node_idx = 2 * current_node_idx + 1
                if debug:
                    print(f"条件 '{condition_tested}' 不满足 (0), 走左子树 -> {next_node_idx}")
                    
            current_node_idx = next_node_idx
        else:
             # Should not happen if tree structure is correct (only C or D roles)
             print(f"Warning: Unknown node role '{current_node.get('role')}' at index {current_node_idx}. Path: {path}. Returning empty class.")
             return tuple([])

def count_internal_nodes(tree):
    """
    计算树的内部节点数量
    
    Args:
        tree: BFS格式的树
    
    Returns:
        内部节点数量
    """
    count = 0
    for node in tree:
        if node is not None and node["role"] == "C":
            count += 1
    return count

def evaluate_trees(trees_json_path, data_csv, class_mapping_json):
    """
    评估多棵树的性能和与原始树的相似度
    
    Args:
        trees_json_path: 包含BFS格式树列表的JSON文件路径
        data_csv: 样本数据CSV文件路径
        class_mapping_json: 类别映射JSON文件路径
    
    Returns:
        每棵树的评估结果字典
    """
    # 加载子树 (JSON format)
    try:
        with open(trees_json_path, 'r') as f:
            json_trees = json.load(f)
    except FileNotFoundError:
        print(f"Error: Trees JSON file not found at {trees_json_path}")
        return {}

    if len(json_trees) < 2:
         print("Error: Need at least 2 trees (original unfolded and generated) in the JSON file.")
         return {}

    # --- Similarity Calculation Setup ---
    # Process all JSON trees into numerical format needed by compare_trees
    # We assume process_json_trees handles the JSON loading internally if needed,
    # but here we pass the loaded data. Let's adjust process_json_trees or replicate logic.
    # For now, let's assume we need to call it once.
    # Re-processing logic might be needed if process_json_trees expects a file path.
    # Let's refine this: process_json_trees expects a path, so we call it once.
    # It saves maps by default, which is fine.
    processed_trees_numerical, feature_names, classes_ = process_json_trees(trees_json_path, save_maps=False) # Don't overwrite maps here

    if not processed_trees_numerical or len(processed_trees_numerical) < 2:
        print("Error processing trees into numerical format.")
        return {}

    original_unfolded_tree_numerical = processed_trees_numerical[1] # Index 1 is the target for comparison
    bounds = [(0, 1) for _ in range(len(feature_names))] # Default bounds

    # --- Performance Evaluation Setup ---
    # 加载样本数据
    df = pd.read_csv(data_csv)
    
    # 分离特征和目标
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 加载类别映射
    with open(class_mapping_json, 'r') as f:
        class_mapping = json.load(f)
    
    # 反向映射，从类别元组到索引
    class_to_idx = {}
    for idx, class_list in class_mapping.items():
        class_to_idx[tuple(sorted(class_list))] = int(idx)
    
    # 评估每棵树 (starting from index 1, the original unfolded tree itself)
    results = {}
    
    # Use json_trees for prediction loop, but processed_trees_numerical for similarity
    for tree_idx, json_tree in enumerate(json_trees):
        if tree_idx < 1:  # Skip index 0 (original non-unfolded)
            continue
            
        if not json_tree: # Skip empty trees if any
            print(f"Skipping empty tree at index {tree_idx}")
            continue
        
        # 将样本转换为字典列表，方便预测
        samples = []
        for _, row in X.iterrows():
            sample = row.to_dict()
            samples.append(sample)
        
        # 对每个样本进行预测
        predicted_classes = []
        for sample in samples:
            predicted_class = predict_with_tree(json_tree, sample, debug=False) # Use json_tree for prediction logic
            predicted_idx = class_to_idx.get(predicted_class, -1)  # 如果找不到映射，返回-1
            predicted_classes.append(predicted_idx)
        
        # 评估指标
        accuracy = accuracy_score(y, predicted_classes)
        f1_macro = f1_score(y, predicted_classes, average='macro', zero_division=0.0) # Added zero_division
        complexity = count_internal_nodes(json_tree) # Use json_tree for complexity

        # --- Structural Similarity Calculation ---
        current_tree_numerical = processed_trees_numerical[tree_idx]
        structural_similarity = 0.0
        if current_tree_numerical and original_unfolded_tree_numerical: # Check if both trees were processed correctly
            try:
                structural_similarity = compare_trees(
                    tree1=original_unfolded_tree_numerical,
                    tree2=current_tree_numerical,
                    feature_names=feature_names,
                    classes_=classes_,
                    bounds=bounds,
                    comp_dist=True, # Or False based on desired comparison strictness
                    dist_weight=0.5
                )
            except Exception as e:
                print(f"Error calculating similarity for tree {tree_idx}: {e}")
                structural_similarity = -1.0 # Indicate error
        
        # 保存结果
        results[tree_idx] = {
            'accuracy': accuracy,
            'f1_score': f1_macro,
            'complexity': complexity,
            'similarity_to_original': structural_similarity # Added similarity
        }
        
        print(f"树 {tree_idx} 的评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1 分数: {f1_macro:.4f}")
        print(f"  复杂度 (内部节点数): {complexity}")
        print(f"  与原始展开树的结构相似度: {structural_similarity:.4f}") # Added print
        print(f"  分类报告:")
        print(classification_report(y, predicted_classes))
        print()
    
    return results

def main():
    # 设置路径 (These should ideally become arguments)
    data_dir = "data/0503_045341/WD" # Example, make this an argument
    subtrees_path = f"{data_dir}/generated_subtrees.json"
    samples_path = f"{data_dir}/unified_synthetic_samples.csv"
    class_mapping_path = f"{data_dir}/unified_class_mapping.json"
    
    # 加载和处理参数 (Example using argparse)
    # import argparse
    # parser = argparse.ArgumentParser(description="Evaluate generated decision trees.")
    # parser.add_argument("subtrees_path", help="Path to the generated_subtrees.json file.")
    # parser.add_argument("samples_path", help="Path to the unified_synthetic_samples.csv file.")
    # parser.add_argument("class_mapping_path", help="Path to the unified_class_mapping.json file.")
    # parser.add_argument("--output_dir", default=".", help="Directory to save evaluation results.")
    # args = parser.parse_args()
    
    # subtrees_path = args.subtrees_path
    # samples_path = args.samples_path
    # class_mapping_path = args.class_mapping_path
    # output_dir = args.output_dir
    
    # --- Example paths used for now ---

    if not all(os.path.exists(p) for p in [subtrees_path, samples_path, class_mapping_path]):
        print("Error: One or more input files not found.")
        print(f" Checked: {subtrees_path}, {samples_path}, {class_mapping_path}")
        return
    
    # 评估树
    results = evaluate_trees(subtrees_path, samples_path, class_mapping_path)

    if not results:
        print("No evaluation results generated.")
        return
    
    # 保存结果 (Save relative to the input data or specified output dir)
    output_dir = os.path.dirname(subtrees_path) # Save in the same dir as subtrees
    output_path = os.path.join(output_dir, "tree_evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存到 {output_path}")
    
    # 找出最佳树 (e.g., based on F1 score)
    if results:
        best_tree_idx = max(results.keys(), key=lambda k: results[k]['f1_score'])
        print(f"最佳树 (按F1分数): 树 {best_tree_idx}")
        print(f"  Accuracy: {results[best_tree_idx]['accuracy']:.4f}")
        print(f"  F1 Score: {results[best_tree_idx]['f1_score']:.4f}")
        print(f"  Complexity: {results[best_tree_idx]['complexity']}")
        print(f"  Similarity: {results[best_tree_idx]['similarity_to_original']:.4f}")
    
    # 计算不同复杂度树的平均性能
    complexity_to_performance = defaultdict(lambda: {'f1_scores': [], 'similarities': []})
    for tree_idx, metrics in results.items():
        complexity = metrics['complexity']
        complexity_to_performance[complexity]['f1_scores'].append(metrics['f1_score'])
        complexity_to_performance[complexity]['similarities'].append(metrics['similarity_to_original'])
    
    print("不同复杂度树的平均性能和相似度:")
    for complexity, data in sorted(complexity_to_performance.items()):
        avg_f1 = sum(data['f1_scores']) / len(data['f1_scores'])
        avg_sim = sum(s for s in data['similarities'] if s != -1.0) / len([s for s in data['similarities'] if s != -1.0]) if any(s != -1.0 for s in data['similarities']) else 0.0
        num_trees = len(data['f1_scores'])
        print(f"  复杂度 {complexity}: 平均F1={avg_f1:.4f}, 平均相似度={avg_sim:.4f} (基于 {num_trees} 棵树)")

if __name__ == "__main__":
    main() 