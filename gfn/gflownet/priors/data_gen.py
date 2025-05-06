import json
import numpy as np
import pandas as pd
import random
import os
from collections import deque, defaultdict
import argparse

def extract_features_and_classes(tree):
    """
    从树中提取所有特征和类别
    
    Args:
        tree: 树的BFS表示
        
    Returns:
        features: 所有特征的集合
        classes: 所有类别的集合
    """
    features = set()
    classes = set()
    
    for node in tree:
        if node is None:
            continue
            
        if node["role"] == "C":
            # 从条件节点中提取特征
            for condition in node["triples"]:
                features.add(condition)
        elif node["role"] == "D":  # 移除了 and node["triples"] 条件，现在空决策节点也会被考虑
            # 从决策节点中提取类别
            class_tuple = tuple(sorted(node["triples"]))
            classes.add(class_tuple)
    
    return features, classes

def find_path_to_leaf(tree, leaf_index):
    """
    找到从根节点到叶节点的路径
    
    Args:
        tree: 树的BFS表示
        leaf_index: 叶节点在BFS表示中的索引
        
    Returns:
        path: 从根节点到叶节点的路径，包含每个节点的索引
    """
    path = []
    current_index = leaf_index
    
    # 从叶节点回溯到根节点
    while current_index > 0:
        path.append(current_index)
        # 计算父节点索引
        parent_index = (current_index - 1) // 2
        current_index = parent_index
    
    # 添加根节点
    path.append(0)
    # 反转路径，使其从根节点开始
    path.reverse()
    
    return path

def get_leaf_nodes_by_class(tree):
    """
    按类别获取所有叶节点
    
    Args:
        tree: 树的BFS表示
        
    Returns:
        leaf_nodes_by_class: 按类别分组的叶节点索引字典
    """
    leaf_nodes_by_class = defaultdict(list)
    
    for i, node in enumerate(tree):
        if node is not None and node["role"] == "D":  # 移除了 and node["triples"] 条件
            class_key = tuple(sorted(node["triples"]))
            leaf_nodes_by_class[class_key].append(i)
    
    return leaf_nodes_by_class

def generate_sample_for_path(tree, path, features):
    """
    For a given path, generate a sample that strictly follows that path.
    Features tested on the path are set deterministically (0 for left/False, 1 for right/True).
    Other features are randomized.

    Args:
        tree: 树的BFS表示
        path: 从根节点到叶节点的路径
        features: 所有特征的列表

    Returns:
        sample: 生成的样本，特征值为0或1
    """
    # Initialize sample with random values first
    sample = {feature: random.randint(0, 1) for feature in features}
    features_tested_on_path = set()

    # Enforce path constraints
    for i in range(len(path) - 1): # Iterate through internal nodes on the path
        current_idx = path[i]
        next_idx_on_path = path[i+1]
        node = tree[current_idx]

        if node is None or node["role"] != "C":
            assert False, f"Node at index {current_idx} is not a condition node."

        conditions = node["triples"]
        # After unfolding, there should only be one condition per C node
        if not conditions:
            print(f"Warning: Condition node {current_idx} has no conditions in path {path}")
            continue
        
        # Assert that there's exactly one condition after unfolding
        assert len(conditions) == 1, f"Node {current_idx} in unfolded tree should have 1 condition, got {len(conditions)}"
        feature_tested = conditions[0]
        
        if feature_tested not in sample:
             print(f"Warning: Feature '{feature_tested}' tested at node {current_idx} not in global feature list for path {path}. Skipping constraint.")
             continue # Skip if the feature isn't in the global list (shouldn't happen ideally)

        features_tested_on_path.add(feature_tested)

        left_child_idx = 2 * current_idx + 1
        right_child_idx = 2 * current_idx + 2

        if next_idx_on_path == left_child_idx:
            # Path goes left (False branch) -> Feature MUST be 0
            sample[feature_tested] = 0
        elif next_idx_on_path == right_child_idx:
            # Path goes right (True branch) -> Feature MUST be 1
            sample[feature_tested] = 1
        else:
             # Should not happen for a valid path
             print(f"Warning: Path deviation detected at node {current_idx} for path {path}")

    # Features not explicitly tested on this path retain their random values from initialization.
    # No additional randomization step is needed here as it was done initially.

    return sample

def generate_samples(tree, num_samples_per_class=100):
    """
    为决策树中的每个类别生成样本

    Args:
        tree: 树的BFS表示
        num_samples_per_class: 每个类别生成的样本数量

    Returns:
        samples: 生成的样本列表
        features: 所有特征的列表
        class_indices: 每个样本对应的类别索引
        classes: 类别列表
    """
    # 提取所有特征和类别
    features_set, classes_set = extract_features_and_classes(tree) # 使用集合避免重复
    all_features_list = sorted(list(features_set)) # 排序以保证顺序
    all_classes_list = list(classes_set) # 获取所有类别


    # 按类别获取叶节点
    leaf_nodes_by_class = get_leaf_nodes_by_class(tree)

    samples = []
    class_indices = []
    global_class_map = {cls: i for i, cls in enumerate(all_classes_list)} # 创建全局类别到索引的映射

    # 为每个类别生成等量的样本
    for class_key, leaf_nodes in leaf_nodes_by_class.items():
        if not leaf_nodes: continue # 如果该类别没有叶节点，则跳过

        class_idx = global_class_map[class_key] # 获取全局类别索引

        # 每个类别生成num_samples_per_class个样本
        samples_generated_for_this_class = 0
        attempts = 0 # 防止无限循环
        max_attempts = num_samples_per_class * len(leaf_nodes) * 2 # 设定一个尝试上限

        while samples_generated_for_this_class < num_samples_per_class and attempts < max_attempts:
            attempts += 1
            # 从该类别的叶节点中随机选择一个
            leaf_index = random.choice(leaf_nodes)

            # 找到从根节点到叶节点的路径
            path = find_path_to_leaf(tree, leaf_index)

            # 生成样本 (使用更新后的逻辑和所有特征列表)
            sample = generate_sample_for_path(tree, path, all_features_list)

            # 验证生成的样本是否真的能到达目标叶节点（可选但推荐）
            # predicted_class_tuple = predict_with_tree(tree, sample) # 需要导入 predict_with_tree 或在此复制其逻辑
            # if tuple(sorted(tree[leaf_index]['triples'] if tree[leaf_index] else [])) == predicted_class_tuple:
            #     samples.append(sample)
            #     class_indices.append(class_idx)
            #     samples_generated_for_this_class += 1
            # else:
            #     # 如果生成的样本未能正确分类，可以选择记录或跳过
            #     # print(f"Warning: Generated sample for path {path} did not predict correctly.")
            #     pass

            # --- 移除验证步骤以匹配原始逻辑，只使用生成函数 ---
            samples.append(sample)
            class_indices.append(class_idx)
            samples_generated_for_this_class += 1
            # --- 移除结束 ---


    # 注意：返回的是全局的特征列表和类别列表
    return samples, all_features_list, class_indices, all_classes_list

def save_samples_to_csv(samples, features, class_indices, output_path):
    """
    保存样本到CSV文件
    
    Args:
        samples: 样本列表
        features: 特征列表
        class_indices: 类别索引列表
        output_path: 输出路径
    """
    # 将样本转换为DataFrame
    df_data = []
    for i, sample in enumerate(samples):
        row = [sample[feature] for feature in features]
        row.append(class_indices[i])  # 添加类别标签
        df_data.append(row)
    
    # 创建列名
    column_names = features + ["target"]
    
    # 创建DataFrame
    df = pd.DataFrame(df_data, columns=column_names)
    
    # 保存为CSV
    df.to_csv(output_path, index=False)

def create_class_mapping(classes, output_path):
    """
    创建类别映射文件
    
    Args:
        classes: 类别列表
        output_path: 输出路径
    """
    class_mapping = {i: list(cls) for i, cls in enumerate(classes)}
    with open(output_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

def main():
    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(description="Generate synthetic data from decision trees.")
    parser.add_argument("subtrees_path", help="Path to the generated_subtrees.json file.")
    parser.add_argument("--output_dir", default=None, help="Directory to save generated data. Defaults to the same directory as subtrees_path.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate per class.")
    parser.add_argument("--skip_indices", type=int, nargs='*', default=[0], help="Indices of trees in the JSON file to skip (e.g., [0] skips the original non-unfolded tree).")
    args = parser.parse_args()

    generated_subtrees_path = args.subtrees_path
    output_dir = args.output_dir
    num_samples_per_class = args.num_samples
    skip_indices = set(args.skip_indices)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(generated_subtrees_path)
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths based on output_dir
    unified_csv_path = os.path.join(output_dir, "unified_synthetic_samples.csv")
    unified_class_mapping_path = os.path.join(output_dir, "unified_class_mapping.json")

    # 加载生成的子树
    try:
        with open(generated_subtrees_path, "r") as f:
            generated_subtrees = json.load(f)
    except FileNotFoundError:
        print(f"Error: Subtrees file not found at {generated_subtrees_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {generated_subtrees_path}")
        return

    # 步骤1: 收集所有树的特征和类别 (保持不变)
    all_features = set()
    all_classes_set = set() # 使用集合

    # 首先遍历所有树，收集所有特征和类别
    for tree_idx, tree in enumerate(generated_subtrees):
        if tree_idx in skip_indices: # Use skip_indices set
            continue
        if not tree: continue # 跳过空树

        # 提取当前树的特征和类别
        tree_features, tree_classes = extract_features_and_classes(tree)

        # 添加到全局集合
        all_features.update(tree_features)
        all_classes_set.update(tree_classes)

    # 将集合转换为有序的列表
    all_features_list = sorted(list(all_features)) # 排序特征

    # 对类别进行排序 (保持不变)
    class_strings = []
    for cls in all_classes_set:
        if not cls:
            class_str = "__EMPTY__"
        else:
            class_str = " ".join(sorted(cls))
        class_strings.append((cls, class_str))

    class_strings.sort(key=lambda x: x[1])
    all_classes_list = [cls for cls, _ in class_strings] # 排序后的类别列表

    # 步骤2: 为每个类别收集所有可能的叶节点和对应的树 (保持不变)
    class_to_tree_nodes = defaultdict(list)
    for tree_idx, tree in enumerate(generated_subtrees):
        if tree_idx in skip_indices:
            continue
        if not tree: continue

        leaf_nodes_by_class = get_leaf_nodes_by_class(tree)
        for class_key, leaf_nodes in leaf_nodes_by_class.items():
            for leaf_idx in leaf_nodes:
                class_to_tree_nodes[class_key].append((tree_idx, leaf_idx))

    # 步骤3: 为每个类别生成样本 (使用更新后的生成逻辑)
    all_samples = []
    all_class_indices = []
    global_class_map = {cls: i for i, cls in enumerate(all_classes_list)} # 全局类别映射

    for class_key, tree_nodes in class_to_tree_nodes.items():
        if not tree_nodes:
            continue

        class_idx = global_class_map[class_key] # 获取全局索引

        # 为该类别生成指定数量的样本
        samples_generated_for_this_class = 0
        attempts = 0
        max_attempts = num_samples_per_class * len(tree_nodes) * 5 # 增加尝试次数

        while samples_generated_for_this_class < num_samples_per_class and attempts < max_attempts:
            attempts += 1
            # 随机选择一个树和叶节点
            tree_idx, leaf_idx = random.choice(tree_nodes)
            tree = generated_subtrees[tree_idx]
            if not tree: continue # 以防万一

            # 找到从根节点到叶节点的路径
            path = find_path_to_leaf(tree, leaf_idx)

            # 生成样本（使用全局特征列表和更新后的逻辑）
            sample = generate_sample_for_path(tree, path, all_features_list)

            all_samples.append(sample)
            all_class_indices.append(class_idx)
            samples_generated_for_this_class += 1


    # 保存统一的样本集 (使用全局特征列表)
    save_samples_to_csv(all_samples, all_features_list, all_class_indices, unified_csv_path)

    # 创建统一的类别映射 (使用全局类别列表)
    create_class_mapping(all_classes_list, unified_class_mapping_path)

    print(f"生成统一数据集:")
    print(f"  样本已保存到 {unified_csv_path}")
    print(f"  类别映射已保存到 {unified_class_mapping_path}")
    print(f"  特征总数: {len(all_features_list)}")
    print(f"  类别总数: {len(all_classes_list)}")
    print(f"  样本总数: {len(all_samples)}")
    print(f"  目标每个类别的样本数: {num_samples_per_class}")
    # Print first few features for brevity
    print(f"  特征顺序 (前5个): {all_features_list[:5]}...")
    print(f"  类别映射: {class_to_string_mapping(all_classes_list)}")

    # 可选: 为单独的树生成样本（逻辑也需要同步更新，但此处省略）
    # ... (kept the same, but note generate_samples now uses the global feature/class list)

def class_to_string_mapping(classes):
    """
    将类别元组列表转换为字符串表示的映射，方便调试
    """
    result = {}
    for i, cls in enumerate(classes):
        if not cls:
            result[i] = "__EMPTY__"
        else:
            result[i] = " ".join(sorted(cls))
    return result
            
if __name__ == "__main__":
    main()
