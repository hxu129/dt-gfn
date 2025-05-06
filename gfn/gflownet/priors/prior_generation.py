#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prior_generation.py - 决策树结构先验生成和评估的集成演示脚本

这个脚本集成了以下功能:
1. 从原始树样本中生成多样化的树变体 (sample.py)
2. 为生成的树创建合成数据集 (data_gen.py)
3. 评估生成树的性能和结构相似度 (tree_evaluation.py)

无需使用命令行参数，所有配置都可以在本脚本中直接指定。
"""

import os
import json
import copy
import numpy as np
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict
import sys

# 导入各个模块的关键功能
# 从 sample.py 导入
from sample import (
    extract_disease_name, 
    convert_to_pointer_tree, 
    unfold_pointer_tree, 
    convert_pointer_tree_to_bfs_tree,
    convert_full_bfs_tree_to_pointer_tree,
    check_tree_quality,
    convert_triplets,
    generate_subtree
)

# 从 data_gen.py 导入
from data_gen import (
    extract_features_and_classes,
    find_path_to_leaf,
    get_leaf_nodes_by_class,
    generate_sample_for_path,
    save_samples_to_csv,
    create_class_mapping,
    class_to_string_mapping
)

# 从 tree_evaluation.py 导入
from tree_evaluation import (
    predict_with_tree,
    count_internal_nodes,
    evaluate_trees
)

# 从 json_tree.py 和 gfn_trees.py 导入 (用于结构相似度计算)
from json_tree import process_json_trees
from gfn_trees import compare_trees

def sample_and_generate_trees(
    dataset_path="all_trees.json",
    num_sample_data=7,
    num_desired_subtrees=10,
    sample_params=None,
    generation_params=None,
    output_dir=None,
    save_original_tree=True,
):
    """
    从原始树集合中采样，并生成多样化的子树结构
    
    Args:
        dataset_path: 原始树数据集的路径
        num_sample_data: 要采样的树的数量
        num_desired_subtrees: 每棵采样树生成的子树数量
        sample_params: 树采样参数字典
        generation_params: 子树生成参数字典
        output_dir: 输出目录 (如果为 None，则使用带时间戳的默认目录)
        save_original_tree: 是否保存原始树
        
    Returns:
        dict: 包含生成文件路径的字典
    """
    # 设置默认参数
    if sample_params is None:
        sample_params = {
            'depth_low': 1,      # 从最小深度1开始
            'depth_high': 10,    # 最大深度增至10
            'K_low': 1,          # 最少1个分类
            'K_high': 20,        # 最多20个分类
            'condition_count_low': 1,  # 最少1个条件
            'condition_count_high': 30, # 最多30个条件
        }
    
    if generation_params is None:
        generation_params = {
            'p_swap_children': 0.05,    # 交换子节点
            'p_prune': 0.05,           # 剪枝
            'p_keep_feature': 0.8,     # 保留特征
            'p_add_dummy': 0.05,       # 添加虚拟特征
            'p_add_dummy_parent_layer': 0.05,  # 添加虚拟父层
        }
    
    # 生成输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        base_output_dir = f"data/{timestamp}"
    else:
        base_output_dir = output_dir
    
    # 采样树
    print("正在采样树...")
    print(f"使用数据集: {dataset_path}")
    
    # 去掉try-except，读取原始树数据集
    with open(dataset_path, "r") as f:
        all_trees = json.load(f)
    
    print(f"成功加载数据集，包含 {len(all_trees)} 棵树")
    
    # 1. 采样树
    # 这部分逻辑简化自 sample.py 的 sample_trees 函数
    sampled_trees, disease_names = [], []
    attempts = 0
    max_attempts = sample_params['max_attempts']
    
    print(f"开始采样，目标: {num_sample_data} 棵树，最大尝试次数: {max_attempts}")
    
    while len(set(disease_names)) < num_sample_data and attempts < max_attempts:
        attempts += 1
        print(f"\n--- 尝试 {attempts}/{max_attempts} ---")
        
        # 随机采样一棵树
        original_tree = np.random.choice(all_trees)
        processed_tree = copy.deepcopy(original_tree)
        processed_tree['original_tree'] = original_tree
        new_tree = []

        print(f"处理原始树...")
        # 处理树
        for i, node in enumerate(processed_tree["original_tree"]["tree"]):
            print(f"处理节点 {i}，类型: {node['role'] if isinstance(node, dict) and 'role' in node else 'None'}")
            processed_node = convert_triplets(node)
            new_tree.append(processed_node)

        print("将树转换为指针结构...")
        # 转换为指针结构
        root, nodes = convert_to_pointer_tree(new_tree)

        print("检查树质量...")
        # 检查树质量
        quality_check = check_tree_quality(
            nodes, 
            sample_params['depth_low'], 
            sample_params['depth_high'], 
            sample_params['K_low'], 
            sample_params['K_high'], 
            sample_params['condition_count_low'], 
            sample_params['condition_count_high']
        )
        
        print(f"树质量检查结果: {'通过' if quality_check else '未通过'}")
        
        if not quality_check:
            continue

        # 检查每个条件节点是否有左右子节点
        valid_tree = True
        for node in nodes:
            if node.value['role'] == 'C':
                if node.right is None or node.left is None:
                    print(f"警告: 条件节点没有完整的左右子节点")
                    valid_tree = False
                    break
        
        if not valid_tree:
            continue
        
        print("转换为BFS格式...")
        bfs_tree = convert_pointer_tree_to_bfs_tree(root)
        processed_tree["bfs_tree"] = bfs_tree
        
        print("提取疾病名称...")
        disease_name = extract_disease_name(original_tree)
        print(f"疾病名称: {disease_name}")
        
        if disease_name not in disease_names:
            sampled_trees.append(processed_tree)
            disease_names.append(disease_name)
            print(f"成功采样树 #{len(sampled_trees)}，疾病: {disease_name}")
    
    if not sampled_trees:
        print("警告: 无法采样到符合条件的树")
        return {}
    
    print(f"\n成功采样 {len(sampled_trees)} 棵树，准备生成子树")
    
    # 2. 生成子树
    output_paths = {}
    for sampled_tree, disease_name in zip(sampled_trees, disease_names):
        print(f"\n正在处理疾病: {disease_name}")
        disease_dir = os.path.join(base_output_dir, disease_name)
        os.makedirs(disease_dir, exist_ok=True)
        
        # 保存原始树和展开的版本
        original_tree = sampled_tree["bfs_tree"]
        generated_subtrees = [original_tree]  # 索引0: 原始未展开树
        
        # 展开原始树
        print("展开原始树...")
        original_tree_copy = copy.deepcopy(original_tree)
        root, nodes = convert_full_bfs_tree_to_pointer_tree(original_tree_copy)
        unfold_pointer_tree(root)
        unfolded_tree = convert_pointer_tree_to_bfs_tree(root)
        generated_subtrees.append(unfolded_tree)  # 索引1: 展开的原始树
        
        # 生成更多子树
        print(f"生成 {num_desired_subtrees} 棵子树...")
        for i in range(num_desired_subtrees):
            print(f"生成子树 {i+1}/{num_desired_subtrees}...")
            new_subtree = generate_subtree(original_tree, generation_params)
            generated_subtrees.append(new_subtree)
        
        # 保存生成的树
        subtrees_output_path = os.path.join(disease_dir, "generated_subtrees.json")
        original_tree_output_path = os.path.join(disease_dir, "sampled_tree.json")
        
        print("保存生成的树...")
        with open(subtrees_output_path, "w") as f:
            if save_original_tree:
                json.dump(generated_subtrees, f, indent=2) 
            else:
                json.dump(generated_subtrees[1:], f, indent=2)  # 去掉原始未展开树
        
        with open(original_tree_output_path, "w") as f:
            json.dump(sampled_tree, f, indent=2)
        
        # 绘制树 (仅当 plot_tree 函数可用时)
        print("尝试绘制树...")
        try:
            from graph_viz import plot_tree
            plot_tree(subtrees_output_path)
            print("树绘制成功")
        except Exception as e:
            print(f"树绘制失败: {e}")
        
        # 保存路径信息
        output_paths[disease_name] = {
            'subtrees': os.path.abspath(subtrees_output_path),
            'original_tree': os.path.abspath(original_tree_output_path),
            'directory': os.path.abspath(disease_dir)
        }
    
    print(f"树采样和生成完成。生成了 {len(sampled_trees)} 个疾病的树结构。")
    return output_paths

def generate_synthetic_data(
    subtrees_path,
    output_dir=None,
    num_samples_per_class=100,
    skip_indices=[0] # Note: This still determines which trees are scanned for *features*, but not for generation paths/classes.
):
    """
    为生成的树结构创建合成数据 (修正逻辑：只基于 index=1 生成)

    Args:
        subtrees_path: 包含生成子树的 JSON 文件的路径
        output_dir: 输出目录 (默认为子树文件所在目录)
        num_samples_per_class: 每个类别生成的样本数
        skip_indices: 要跳过的树索引列表 (用于收集 *特征*)

    Returns:
        dict: 包含输出文件路径的字典
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(subtrees_path)
    os.makedirs(output_dir, exist_ok=True)

    # 确定输出文件路径
    unified_csv_path = os.path.join(output_dir, "unified_synthetic_samples.csv")
    unified_class_mapping_path = os.path.join(output_dir, "unified_class_mapping.json")

    # 加载生成的子树
    print(f"加载子树文件: {subtrees_path}")
    with open(subtrees_path, "r") as f:
        generated_subtrees = json.load(f)

    print(f"成功加载 {len(generated_subtrees)} 棵子树")

    if len(generated_subtrees) <= 1:
        print("错误: generated_subtrees.json 需要至少包含索引为 1 的树 (ground truth) 用于生成样本。")
        return {}
        
    ground_truth_tree_idx = 1 # subtree里存了所有树的结构，第1个是unfolded的原始树
    if ground_truth_tree_idx >= len(generated_subtrees) or generated_subtrees[ground_truth_tree_idx] is None:
         print(f"错误: 索引为 {ground_truth_tree_idx} 的 ground truth 树不存在或为空。")
         return {}
    ground_truth_tree = generated_subtrees[ground_truth_tree_idx]

    # 转换跳过索引为集合以提高检查效率 (用于特征收集)
    skip_indices_set = set(skip_indices)

    # 步骤1: 收集所有树的特征 (保持不变，以确保样本包含所有可能的列)
    print("收集所有树的特征...")
    all_features_set = set()
    for tree_idx, tree in enumerate(generated_subtrees):
        if tree_idx in skip_indices_set or tree is None: # 跳过索引和空树
            # print(f"跳过树 {tree_idx} (特征收集)") # 可以取消注释以调试
            continue
        # print(f"处理树 {tree_idx} (特征收集)...") # 可以取消注释以调试
        tree_features, _ = extract_features_and_classes(tree)
        all_features_set.update(tree_features)

    all_features_list = sorted(list(all_features_set))
    print(f"找到 {len(all_features_list)} 个全局特征")

    # 步骤 1.5: 从 Ground Truth Tree (index=1) 提取目标类别
    print(f"从 Ground Truth 树 (index={ground_truth_tree_idx}) 提取目标类别...")
    _, ground_truth_classes_set = extract_features_and_classes(ground_truth_tree)

    # 对目标类别进行排序
    print("排序目标类别...")
    target_class_strings = []
    for cls in ground_truth_classes_set:
        if not cls:
            class_str = "__EMPTY__"
        else:
            class_str = " ".join(sorted(cls))
        target_class_strings.append((cls, class_str))
    target_class_strings.sort(key=lambda x: x[1])
    target_classes_list = [cls for cls, _ in target_class_strings]
    print(f"找到 {len(target_classes_list)} 个目标类别")
    
    if not target_classes_list:
        print(f"错误: Ground Truth 树 (index={ground_truth_tree_idx}) 没有找到任何类别。")
        return {}

    # 步骤2: 只为 Ground Truth Tree (index=1) 收集叶节点
    print(f"只为 Ground Truth 树 (index={ground_truth_tree_idx}) 收集叶节点...")
    gt_leaf_nodes_by_class = get_leaf_nodes_by_class(ground_truth_tree)

    # 步骤3: 只基于 Ground Truth Tree (index=1) 的路径生成样本
    print("基于 Ground Truth 树生成样本...")
    all_samples = []
    all_class_indices = []
    target_class_map = {cls: i for i, cls in enumerate(target_classes_list)} # 目标类别映射

    for class_key, leaf_indices in gt_leaf_nodes_by_class.items():
        if class_key not in target_class_map: # 跳过不在目标类别列表中的类别（理论上不应发生）
             print(f"警告: 在 GT 树叶节点中找到非目标类别 {class_key}，跳过。")
             continue
             
        if not leaf_indices: # 如果该类别没有叶节点（理论上不应发生）
            print(f"警告: 目标类别 {class_key} 在 GT 树中没有找到叶节点，跳过。")
            continue

        class_idx = target_class_map[class_key] # 获取目标类别索引
        print(f"为目标类别 {class_idx} ({class_key}) 生成 {num_samples_per_class} 个样本...")

        # 为该类别生成指定数量的样本
        for i in range(num_samples_per_class):
            if i % 20 == 0 and i > 0: # 避免打印 0/N
                print(f"  已生成 {i}/{num_samples_per_class} 个样本")

            # 从 GT 树的该类别的叶节点中随机选择一个
            leaf_idx = random.choice(leaf_indices)

            # 在 GT 树中找到从根节点到叶节点的路径
            path = find_path_to_leaf(ground_truth_tree, leaf_idx)

            # 在 GT 树上基于该路径生成样本（使用全局特征列表）
            sample = generate_sample_for_path(ground_truth_tree, path, all_features_list)

            all_samples.append(sample)
            all_class_indices.append(class_idx)
        print(f"  已生成 {num_samples_per_class}/{num_samples_per_class} 个样本")


    # 保存统一的样本集 (使用全局特征列表)
    print("保存样本集...")
    save_samples_to_csv(all_samples, all_features_list, all_class_indices, unified_csv_path)

    # 创建统一的类别映射 (只使用来自 GT 树的目标类别)
    print("创建类别映射...")
    create_class_mapping(target_classes_list, unified_class_mapping_path)

    print(f"合成数据生成完成:")
    print(f"  样本已保存到 {unified_csv_path}")
    print(f"  类别映射已保存到 {unified_class_mapping_path}")
    print(f"  特征总数 (来自所有树): {len(all_features_list)}")
    print(f"  类别总数 (来自 GT 树): {len(target_classes_list)}")
    print(f"  样本总数: {len(all_samples)}")
    print(f"  每个目标类别的样本数: {num_samples_per_class}")

    return {
        'samples_csv': os.path.abspath(unified_csv_path),
        'class_mapping': os.path.abspath(unified_class_mapping_path)
    }

def evaluate_tree_performance(
    subtrees_path, 
    samples_csv_path, 
    class_mapping_path
):
    """
    评估生成的树结构的性能和与原始树的相似度
    
    Args:
        subtrees_path: 包含生成子树的 JSON 文件的路径
        samples_csv_path: 合成样本的 CSV 文件路径
        class_mapping_path: 类别映射的 JSON 文件路径
        
    Returns:
        dict: 包含评估结果和输出文件路径的字典
    """
    # 检查输入文件是否存在
    input_files = [subtrees_path, samples_csv_path, class_mapping_path]
    for f in input_files:
        if not os.path.exists(f):
            print(f"错误: 文件不存在: {f}")
    
    if not all(os.path.exists(p) for p in input_files):
        missing = [p for p in input_files if not os.path.exists(p)]
        print(f"错误: 以下输入文件不存在: {', '.join(missing)}")
        return {}
    
    # 确定输出文件路径
    output_dir = os.path.dirname(subtrees_path)
    output_path = os.path.join(output_dir, "tree_evaluation_results.json")
    
    # 评估树
    print("开始评估树性能...")
    results = evaluate_trees(subtrees_path, samples_csv_path, class_mapping_path)
    
    if not results:
        print("未生成评估结果。")
        return {}
    
    # 保存结果
    print(f"保存评估结果到 {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存到 {output_path}")
    
    # 找出最佳树
    if results:
        keys = list(results.keys())
        keys.remove(1) # 移除原始树
        best_tree_idx = max(keys, key=lambda k: results[k]['f1_score'])
        print(f"\n最佳树 (按F1分数): 树 {best_tree_idx}")
        print(f"  准确率: {results[best_tree_idx]['accuracy']:.4f}")
        print(f"  F1分数: {results[best_tree_idx]['f1_score']:.4f}")
        print(f"  复杂度: {results[best_tree_idx]['complexity']}")
        print(f"  相似度: {results[best_tree_idx]['similarity_to_original']:.4f}")
    
    # 分析不同复杂度树的性能
    complexity_to_performance = defaultdict(lambda: {'f1_scores': [], 'similarities': []})
    for tree_idx, metrics in results.items():
        complexity = metrics['complexity']
        complexity_to_performance[complexity]['f1_scores'].append(metrics['f1_score'])
        complexity_to_performance[complexity]['similarities'].append(metrics['similarity_to_original'])
    
    print("\n不同复杂度树的平均性能和相似度:")
    for complexity, data in sorted(complexity_to_performance.items()):
        if data['f1_scores']:
            avg_f1 = sum(data['f1_scores']) / len(data['f1_scores'])
            # 计算平均相似度，忽略错误值（-1.0）
            valid_sims = [s for s in data['similarities'] if s != -1.0]
            avg_sim = sum(valid_sims) / len(valid_sims) if valid_sims else 0.0
            num_trees = len(data['f1_scores'])
            print(f"  复杂度 {complexity}: 平均F1={avg_f1:.4f}, 平均相似度={avg_sim:.4f} (基于 {num_trees} 棵树)")
    
    return {
        'results': results,
        'results_json': os.path.abspath(output_path),
        'best_tree_idx': best_tree_idx if results else None
    }

def create_structural_prior(
    subtrees_path,
    not_selected_indices=None,
    output_file=None
):
    """
    从生成的子树中选择一部分作为结构先验，并保存为新的 JSON 文件
    
    Args:
        subtrees_path: 包含生成子树的 JSON 文件的路径
        not_selected_indices: 要跳过的树索引列表，None 表示选择所有索引 >= 1 的树
        output_file: 输出文件路径，None 表示在原目录创建 "structural_priors.json"
        
    Returns:
        str: 结构先验文件的路径
    """
    # 加载子树
    print(f"读取子树文件: {subtrees_path}")
    with open(subtrees_path, 'r') as f:
        all_trees = json.load(f)
    
    # 选择指定的树
    selected_trees = [all_trees[i] for i in range(len(all_trees)) if i not in not_selected_indices]
    
    if not selected_trees:
        print("警告: 没有选择任何树作为结构先验")
        return None
    
    # 确定输出文件路径
    if output_file is None:
        output_dir = os.path.dirname(subtrees_path)
        output_file = os.path.join(output_dir, "structural_priors.json")
    
    # 保存选定的树
    print(f"保存 {len(selected_trees)} 棵树到 {output_file}")
    with open(output_file, 'w') as f:
        json.dump(selected_trees, f, indent=2)
    
    print(f"已选择 {len(selected_trees)} 棵树作为结构先验")
    print(f"结构先验已保存到 {output_file}")
    
    return os.path.abspath(output_file)

def calculate_average_similarity(
    new_tree_numerical,
    priors_json,
    comp_dist=True,
    dist_weight=0.5,
    bounds=None
):
    """
    计算新树（数值格式）与结构先验集的平均相似度

    Args:
        new_tree_numerical (list or np.ndarray): 新树，格式为数值列表的列表 (gfn_trees.py main中的格式).
                                                 **重要**: 此树必须使用与 priors_json 文件
                                                 相同的特征和类别映射进行编码。
        priors_json (str): 结构先验的 JSON 文件路径.
        comp_dist (bool): 是否比较标签分布.
        dist_weight (float): 标签分布差异的权重.
        bounds (list, optional): 特征的边界列表，例如 [(0, 1), (0, 1), ...]。
                                 如果为 None，将在 compare_trees 中使用默认值。

    Returns:
        float: 与结构先验的平均相似度. 返回 0.0 如果无法计算（例如先验文件无效）。
    """
    from json_tree import create_similarity_calculator

    # 创建相似度计算器 (这会处理先验树的加载和数值转换)
    print(f"创建相似度计算器，使用先验文件: {priors_json}")
    try:
        similarity_calculator = create_similarity_calculator(priors_json, save_maps=False)
    except FileNotFoundError:
        print(f"错误: 先验文件 {priors_json} 未找到。")
        return 0.0
    except Exception as e:
        print(f"错误: 创建相似度计算器时出错: {e}")
        return 0.0

    # 直接使用预处理的 new_tree_numerical 调用计算器
    print("计算与先验树的平均相似度...")
    try:
        similarity = similarity_calculator(
            input_tree=new_tree_numerical,
            comp_dist=comp_dist,
            dist_weight=dist_weight, 
            bounds=bounds
        )
        print(f"平均相似度计算结果: {similarity:.4f}")
        return similarity
    except Exception as e:
        print(f"错误: 计算相似度时出错: {e}")
        return 0.0

def run_complete_pipeline(
    dataset_path="all_trees.json",
    output_dir=None,
    num_sample_data=1,
    num_desired_subtrees=5,
    sample_params=None,
    generation_params=None,
    num_samples_per_class=100,
    skip_indices=[0],
    create_priors=True,
    not_selected_prior_indices=None
):
    """
    运行完整的结构先验生成和评估流程
    
    Args:
        dataset_path: 原始树数据集路径
        output_dir: 输出目录
        num_sample_data: 要采样的树数量
        num_desired_subtrees: 每棵采样树生成的子树数量
        sample_params: 树采样参数
        generation_params: 子树生成参数
        num_samples_per_class: 每个类别生成的样本数
        skip_indices: 数据生成时要跳过的树索引
        create_priors: 是否创建结构先验
        not_selected_prior_indices: 作为结构先验的树索引
        
    Returns:
        dict: 包含流程所有输出的字典
    """
    results = {
        'trees': {},
        'data': {},
        'evaluation': {},
        'priors': {}
    }
    
    # 第1步：采样和生成树
    print("\n=== 步骤 1: 采样和生成树 ===")
    print(f"数据集: {dataset_path}")
    
    # 检查数据集文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件 {dataset_path} 不存在")
        return results
        
    tree_paths = sample_and_generate_trees(
        dataset_path=dataset_path,
        num_sample_data=num_sample_data,
        num_desired_subtrees=num_desired_subtrees,
        sample_params=sample_params,
        generation_params=generation_params,
        output_dir=output_dir
    )
    
    if not tree_paths:
        print("错误: 树生成失败，流程终止")
        return results
    
    results['trees'] = tree_paths
    
    # 处理每个疾病的树
    for disease_name, paths in tree_paths.items():
        subtrees_path = paths['subtrees']
        
        # 第2步：生成合成数据
        print(f"\n=== 步骤 2: 为疾病 '{disease_name}' 生成合成数据 ===")
        data_paths = generate_synthetic_data(
            subtrees_path=subtrees_path,
            num_samples_per_class=num_samples_per_class,
            skip_indices=skip_indices
        )
        
        if not data_paths:
            print(f"警告: 为疾病 '{disease_name}' 生成数据失败，跳过后续步骤")
            continue
        
        results['data'][disease_name] = data_paths
        
        # 第3步：评估树性能
        print(f"\n=== 步骤 3: 评估疾病 '{disease_name}' 的树性能 ===")
        eval_results = evaluate_tree_performance(
            subtrees_path=subtrees_path,
            samples_csv_path=data_paths['samples_csv'],
            class_mapping_path=data_paths['class_mapping']
        )
        
        if eval_results:
            results['evaluation'][disease_name] = eval_results
        
        # 第4步：创建结构先验
        if create_priors:
            print(f"\n=== 步骤 4: 为疾病 '{disease_name}' 创建结构先验 ===")
            priors_path = create_structural_prior(
                subtrees_path=subtrees_path,
                not_selected_indices=not_selected_prior_indices
            )
            
            if priors_path:
                results['priors'][disease_name] = priors_path
    
    print("\n=== 完整流程执行完毕 ===")
    return results

def main():
    """
    演示主函数，使用示例参数运行完整流程
    """
    # 设置更详细的调试输出
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 示例参数配置
    params = {
        'dataset_path': "all_trees.json",  # 原始树数据集路径
        'output_dir': None,  # 自动生成带时间戳的目录
        'num_sample_data': 7,  # 采样树的数量（此处设为1以加快演示）
        'num_desired_subtrees': 100,  # 每棵采样树生成的子树数量
        
        # 树采样参数 - 更宽松的参数，以便通过质量检查
        'sample_params': {
            'depth_low': 3,      # 从最小深度2开始
            'depth_high': 10,    # 最大深度增至10
            'K_low': 3,          # 最少3个分类
            'K_high': 5,        # 最多5个分类
            'condition_count_low': 8,  # 最少4个条件
            'condition_count_high': 100, # 最多5个条件
            'max_attempts': 400, # 最大尝试次数
        },
        
        # 子树生成参数
        'generation_params': {
            'p_swap_children': 0.05,    # 交换子节点概率
            'p_deactivate': 0.07,           # 失活概率
            'p_prune': 0.1, # 剪枝概率
            'p_keep_feature': 0.75,     # 保留特征概率
            'p_add_dummy': 0.01,       # 添加虚拟特征概率
            'p_add_dummy_parent_layer': 0.01,  # 添加虚拟父层概率
        },
        
        # 数据生成参数
        'num_samples_per_class': 5,  # 每个类别生成的样本数
        'skip_indices': [0],  # 数据生成时要跳过的树索引，在这里只有idx=0是原始树，不用来做任何事情
        
        # 结构先验参数
        'create_priors': True,  # 是否创建结构先验文件
        'not_selected_prior_indices': [0, 1],  # 跳过idx=0 (原始树) 和 idx=1 (原始树展开) 作为结构先验
    }
    
    # 检查数据集文件是否存在
    if not os.path.exists(params['dataset_path']):
        print(f"错误: 数据集文件 {params['dataset_path']} 不存在")
        print("请确保 all_trees.json 文件存在，或修改 'dataset_path' 参数指向正确的文件")
        alternate_file = os.path.join(os.getcwd(), "all_trees.json")
        print(f"尝试查找: {alternate_file}")
        if os.path.exists(alternate_file):
            print(f"找到备用文件，使用: {alternate_file}")
            params['dataset_path'] = alternate_file
        else:
            # 尝试查找项目中的示例JSON文件
            print("尝试查找项目中的示例JSON文件...")
            for root, dirs, files in os.walk(os.getcwd()):
                for file in files:
                    if file.endswith('.json'):
                        print(f"找到可能的JSON文件: {os.path.join(root, file)}")
            return
    
    # 运行完整流程
    print("\n开始运行完整流程...")
    results = run_complete_pipeline(**params)
    
    # 打印完整流程的结果汇总
    print("\n=== 结果汇总 ===")
    for disease_name in results['trees'].keys():
        print(f"\n疾病: {disease_name}")
        
        # 树生成结果
        if disease_name in results['trees']:
            print(f"生成的子树: {results['trees'][disease_name]['subtrees']}")
            print(f"原始采样树: {results['trees'][disease_name]['original_tree']}")
        
        # 数据生成结果
        if disease_name in results['data']:
            print(f"合成样本: {results['data'][disease_name]['samples_csv']}")
            print(f"类别映射: {results['data'][disease_name]['class_mapping']}")
        
        # 评估结果
        if disease_name in results['evaluation'] and 'best_tree_idx' in results['evaluation'][disease_name]:
            best_idx = results['evaluation'][disease_name]['best_tree_idx']
            # 处理 best_idx 可能是整数或字符串的情况
            best_key = str(best_idx)  # 首先尝试字符串键
            results_dict = results['evaluation'][disease_name]['results']
            
            if best_key in results_dict:
                best_metrics = results_dict[best_key]
            elif best_idx in results_dict:  # 如果字符串键不存在，尝试整数键
                best_metrics = results_dict[best_idx]
            else:
                print(f"警告: 无法找到树 {best_idx} 的评估结果")
                continue
                
            print(f"最佳树 (索引 {best_idx}): 准确率 = {best_metrics['accuracy']:.4f}, F1 = {best_metrics['f1_score']:.4f}, 相似度 = {best_metrics['similarity_to_original']:.4f}")
        
        # 结构先验
        if disease_name in results['priors']:
            print(f"结构先验: {results['priors'][disease_name]}")
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()
