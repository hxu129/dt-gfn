import numpy as np
import random
import json
import copy
from collections import deque
from graph_viz import plot_tree
from datetime import datetime
import os
from typing import Dict, List, Any
num_of_dummy_features = 0

class Node:
    def __init__(self, value, left=None, right=None, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent

def extract_disease_name(tree):
    """Step 8: Extract disease name for naming the tree"""
    for node in tree["tree"]:
        if node["role"] == "C" and "triples" in node and node["triples"]:
            disease_text = node["triples"][0][0]
            # Extract disease name from patterns like "Patients with X"
            if "Patients with " in disease_text:
                return disease_text.split("Patients with ")[1].strip()
            elif "patients with " in disease_text:
                return disease_text.split("patients with ")[1].strip()
            else:
                return disease_text
    return "unknown_disease"

def convert_to_pointer_tree(tree_data: List[Dict[str, Any]]) -> List[Node]:
    """
    将以BFS列表表示的树（非完整）转换为指针连接的节点列表。

    Args:
        tree_data: 包含键 "tree" 的字典，其值为一个节点字典列表，
                   按BFS顺序排列。每个节点字典至少包含 "role" ('C' 或 'D')。

    Returns:
        一个Node对象的列表，节点间的 left 和 right 指针已根据BFS顺序正确设置。
        如果输入为空或无效，则返回空列表。
    """
    node_dicts: List[Dict[str, Any]] = tree_data
    
    # 1. 创建所有节点对象，保持原始BFS顺序
    nodes: List[Node] = [Node(d) for d in node_dicts]
    num_nodes: int = len(nodes)
    
    # 2. 使用一个索引来跟踪下一个应该被分配为子节点的节点
    #    根节点 (nodes[0]) 的子节点从 nodes[1] 开始
    child_index: int = 1 

    # 3. 遍历所有节点，为内部节点 ('C') 分配子节点
    for i in range(num_nodes):
        current_node = nodes[i]
        
        # 如果当前节点是内部节点，尝试分配子节点
        if current_node.value.get("role") == "C":
            # 分配左子节点 (如果还有未分配的节点)
            if child_index < num_nodes:
                current_node.left = nodes[child_index]
                child_index += 1
            else:
                # 如果需要，可以在这里添加错误处理或日志
                # print(f"Warning: Node at index {i} (Role 'C') expected a left child, but no more nodes available.")
                pass # BFS序列可能在此结束

            # 分配右子节点 (如果还有未分配的节点)
            if child_index < num_nodes:
                current_node.right = nodes[child_index]
                child_index += 1
            else:
                # print(f"Warning: Node at index {i} (Role 'C') expected a right child, but no more nodes available.")
                pass # BFS序列可能在此结束
                
        # 如果当前节点是叶节点 ('D')，它没有子节点，我们什么都不做
        
        # 优化：如果所有后续节点都已被分配为子节点，可以提前停止
        if child_index >= num_nodes:
            break
            
    # 返回包含所有连接好的节点的列表
    # 注意：列表中的第一个节点 nodes[0] 是树的根节点
    return nodes[0], nodes


def convert_full_bfs_tree_to_pointer_tree(bfs_tree):
    """Convert a full bfs_tree to a pointer tree"""
    nodes = [Node(node) for node in bfs_tree]
    for i, node in enumerate(nodes):
        if node.value is None:
            continue
        if 2 * i + 1 < len(nodes):
            if nodes[2 * i + 1].value is not None:
                node.left = nodes[2 * i + 1]
                nodes[2 * i + 1].parent = node
            else:
                node.left = None
        if 2 * i + 2 < len(nodes):
            if nodes[2 * i + 2].value is not None:
                node.right = nodes[2 * i + 2]
                nodes[2 * i + 2].parent = node
            else:
                node.right = None
    return nodes[0], [node for node in nodes if node.value is not None]

def copy_node(node, parent=None):
    new_node = Node(
        {
            'role': node.value['role'],
            'triples': node.value['triples'],
            'logical_rel': node.value['logical_rel']
        }
    )
    if node.left is not None:
        new_node.left = copy_node(node.left, new_node)
    if node.right is not None:
        new_node.right = copy_node(node.right, new_node)
    new_node.parent = parent
    return new_node

def unfold_pointer_tree(root):
    """Unfold a pointer tree using pre-order traversal and careful parent handling."""
    # 如果节点为空、节点值为空或是决策节点 'D'，则停止递归
    if root is None or root.value is None or root.value['role'] == 'D':
        return

    # 标记此节点是否在本轮被展开
    node_was_unfolded = False
    # --- 前序遍历：先处理当前节点 ---
    if root.value['role'] == 'C' and 'triples' in root.value and len(root.value['triples']) > 1:
        node_was_unfolded = True
        conditions = sorted(root.value['triples'])
        original_logical_rel = root.value['logical_rel'] # 保存原始逻辑关系

        # 保存原始的左右子节点引用
        original_left = root.left
        original_right = root.right

        # 创建新节点用于存放剩余的条件 (conditions[1:])
        # 子节点暂时设为 None，稍后根据 'and'/'or' 逻辑设置
        # 父节点直接设置为当前的 root
        new_node = Node(
            value={
                'role': 'C',
                'triples': conditions[1:],
                'logical_rel': original_logical_rel # 新节点初始继承父节点的逻辑关系
            },
            left=None,
            right=None,
            parent=root # new_node 的父节点是 root
        )

        # 更新当前 root 节点，只保留第一个条件
        root.value['triples'] = [conditions[0]]
        # root 的 logical_rel 将在递归结束后根据情况设置为 'null'

        # 左节点是 False，右节点是 True
        if original_logical_rel == 'or':
            # 展开 'or' 节点: (A or B) -> If A is False (Left), evaluate B. If A is True (Right), result is original True branch.
            # root 结构: left=new_node, right=original_right
            # new_node 结构: left=original_left, right=copy(original_right)

            root.left = new_node # root 的左子节点变为 new_node (evaluate B if A is False)

            # new_node 的子节点设置为：左边是 original_left，右边是 original_right 的拷贝
            new_node.left = original_left
            if new_node.left: # 如果 original_left 存在，更新其父节点为 new_node
                new_node.left.parent = new_node
            new_node.right = copy_node(original_right, new_node) # 拷贝右子树，父节点设为 new_node

            # root 的右子节点保持不变 (original_right)，确保其父指针仍指向 root (result is original True branch if A is True)
            if root.right != original_right: root.right = original_right # 安全检查
            if root.right:
                 root.right.parent = root

        elif original_logical_rel == 'and':
            # 展开 'and' 节点: (A and B) -> If A is False (Left), result is original False branch. If A is True (Right), evaluate B.
            # root 结构: left=original_left, right=new_node
            # new_node 结构: left=copy(original_left), right=original_right

            root.right = new_node # root 的右子节点变为 new_node (evaluate B if A is True)

            # new_node 的子节点设置为：左边是 original_left 的拷贝，右边是 original_right
            new_node.left = copy_node(original_left, new_node) # 拷贝左子树，父节点设为 new_node
            new_node.right = original_right
            if new_node.right: # 如果 original_right 存在，更新其父节点为 new_node
                new_node.right.parent = new_node

            # root 的左子节点保持不变 (original_left)，确保其父指针仍指向 root (result is original False branch if A is False)
            if root.left != original_left: root.left = original_left # 安全检查
            if root.left:
                 root.left.parent = root
        else:
            # 如果存在多个 triple 但逻辑关系不是 'and' 或 'or'，则抛出错误
            raise ValueError(f"Invalid logical relation '{original_logical_rel}' for node with multiple triples")

    # --- 递归处理子节点 ---
    # 注意：获取的是可能已经被修改过的子节点引用
    current_left = root.left
    current_right = root.right
    unfold_pointer_tree(current_left)
    unfold_pointer_tree(current_right)

    # --- 后续处理 ---
    # 检查当前节点：如果它仍然是 'C' 节点且只包含一个 triple（无论最初只有一个还是展开后剩一个）
    # 则将其 logical_rel 设置为 'null'
    if root.value['role'] == 'C' and 'triples' in root.value and len(root.value['triples']) == 1:
         root.value['logical_rel'] = 'null'


def convert_pointer_tree_to_bfs_tree(root):
    """Convert a pointer tree to a bfs tree"""
    def get_depth(node):
        if node is None:
            return 0
        return 1 + max(get_depth(node.left), get_depth(node.right))
    
    depth = get_depth(root)
    bfs_tree = [None] * (2 ** depth - 1)

    def fill_bfs_tree(node, index):
        if node is None:
            return
        bfs_tree[index] = node.value
        fill_bfs_tree(node.left, 2 * index + 1)
        fill_bfs_tree(node.right, 2 * index + 2)
    fill_bfs_tree(root, 0)
    while bfs_tree[-1] is None:
        bfs_tree.pop()
    return bfs_tree

def check_or_relationship(nodes):
    """Check if the tree has an "or" relationship in the decision nodes"""
    for node in nodes:
        if node.value["role"] == "D" and node.value["logical_rel"] == "or":
            return False
    return True

def count_classifications(nodes):
    """Count the number of classifications in the tree"""
    return len(set([" ".join(node.value["triples"]) for node in nodes if node.value["role"] == "D"]))

def count_conditions(nodes):
    """Count the number of conditions in the tree"""
    return sum([len(node.value["triples"]) for node in nodes if node.value["role"] == "C"])

def count_depth(nodes):
    """Count the depth of the tree"""
    def helper(node, depth):
        if node is None:
            return depth
        return max(helper(node.left, depth + 1), helper(node.right, depth + 1))
    return helper(nodes[0], 0)

def check_tree_quality(nodes, depth_low, depth_high, K_low, K_high, condition_count_low, condition_count_high):
    """Check if the tree has a good quality"""
    if not check_or_relationship(nodes):
        return False
    if not (depth_low <= count_depth(nodes) <= depth_high):
        return False
    if not (K_low <= count_classifications(nodes) <= K_high):
        return False
    if not (condition_count_low <= count_conditions(nodes) <= condition_count_high):
        return False
    return True


def convert_triplets(node):
    """Step 3: Convert triplets based on the rules"""
    def process_triplet(triple):
        """Process a triplet to extract the feature"""
        if len(triple) < 3:
            return ""
    
        # Concatenate the second and third elements as the feature
        return triple[1] + " " + triple[2]
    if node["role"] == "C":
        # For conditional nodes, extract feature from triplet
        if node["triples"]:
            features = []
            for triple in node["triples"]:
                feature = process_triplet(triple)
                features.append(feature)
            # Keep triples for unfold_internal_nodes to use
            return {"role": "C", "triples": features, "logical_rel": node["logical_rel"]}
    elif node["role"] == "D":
        # For decision nodes, process based on rules
        converted_triples = []
        for triple in node["triples"]:
            if len(triple) < 3:
                continue
                
            if "patient" in triple[0].lower() or "child" in triple[0].lower():
                # Replace with concatenation of second and third strings
                converted_triples.append(triple[1] + " " + triple[2])
            else:
                # Default case, concatenate all elements
                converted_triples.append(" ".join(triple))
        
        return {"role": "D", "triples": converted_triples, "logical_rel": node["logical_rel"]}
    
    return node


def sample_trees(dataset_path, num_desired_subtrees, sample_params):
    """Sample a subtree from the original tree"""
    with open(dataset_path, "r") as f:
        all_trees = json.load(f)
    sampled_trees, disease_names = [], []
    attempts = 0
    max_attempts = 20
    while len(set(disease_names)) < num_desired_subtrees and attempts < max_attempts:
        # Randomly sample a tree
        original_tree = np.random.choice(all_trees)
        processed_tree = copy.deepcopy(original_tree)
        processed_tree['original_tree'] = original_tree
        new_tree = []

        # Process the tree
        for i, node in enumerate(processed_tree["original_tree"]["tree"]):
            processed_node = convert_triplets(node)
            new_tree.append(processed_node)

        # Convert to bfs format
        root, nodes = convert_to_pointer_tree(new_tree)

        # Check tree quality
        if not check_tree_quality(nodes, sample_params['depth_low'], sample_params['depth_high'], sample_params['K_low'], sample_params['K_high'], sample_params['condition_count_low'], sample_params['condition_count_high']):
            continue

        for node in nodes:
            if node.value['role'] == 'C':
                assert node.right is not None, "node.right is None"
                assert node.left is not None, "node.left is None"
        bfs_tree = convert_pointer_tree_to_bfs_tree(root)
        processed_tree["bfs_tree"] = bfs_tree
        disease_name = extract_disease_name(original_tree)
        if disease_name not in disease_names:
            sampled_trees.append(processed_tree)
            disease_names.append(disease_name)
        attempts += 1
    return sampled_trees, disease_names

def swap_children(node):
    """Swap the left and right children of a node"""
    node.left, node.right = node.right, node.left

def deactivate_node(node):
    """Deactivate a node and convert it to a D node (majority sampling)"""
    # FIXME: modify the probability of the children to be D nodes
    def get_children_D_node_values(node) -> list[list[str]]:
        if node is None:
            return []
        if node.value['role'] == 'D':
            return [node.value['triples']]
        else:
            return get_children_D_node_values(node.left) + get_children_D_node_values(node.right)

    if node.value['role'] == 'C':
        children_D_node_values = get_children_D_node_values(node)
        D_values = children_D_node_values[np.random.randint(0, len(children_D_node_values))]
        node.value['triples'] = D_values
        node.value['role'] = 'D'
        children = [node.left, node.right]
        children[0].parent, children[1].parent = None, None
        node.left, node.right = None, None
    elif node.value['role'] == 'D':
        node.value['triples'] = []
    else:
        raise ValueError(f"Invalid node role: {node.value['role']}")

def sample_condition_set(node, p_keep_feature):
    """Sample a subset of conditions from the node"""
    assert node.value['role'] == 'C', "node is not a C node"
    assert node.value['logical_rel'] != 'null', "node is a null logical_rel"
    assert node is not None, "node is None"
    all_conditions = node.value['triples']
    new_conditions = []
    for condition in all_conditions:
        if random.random() < p_keep_feature:
            new_conditions.append(condition)
    if len(new_conditions) == 0:
        # Convert to a sample D node
        deactivate_node(node)
    elif len(new_conditions) == 1:
        node.value['logical_rel'] = 'null'
        node.value['triples'] = new_conditions
    else:
        node.value['triples'] = new_conditions

def add_dummy_feature(node):
    """Add a dummy feature to the node"""
    assert node.value['role'] == 'C', "node is not a C node, but a " + node.value['role']
    assert node is not None, "node is None"
    global num_of_dummy_features
    node.value['triples'].append(f"dummy_feature_{num_of_dummy_features}")
    if node.value['logical_rel'] == 'null':
        node.value['logical_rel'] = 'and' if random.random() < 0.5 else 'or'
    num_of_dummy_features += 1

def add_dummy_parent_layer(node):
    """Add a dummy parent layer to the node"""
    assert node is not None, "node is None"
    parent = node.parent
    if parent is None:
        return
    # To get whether this is left child or right child
    is_left_child = parent.left == node
    global num_of_dummy_features
    new_node = Node(value={'role': 'C', 'triples': [f'dummy_feature_{num_of_dummy_features}'], 'logical_rel': 'null'})
    num_of_dummy_features += 1
    if is_left_child:
        parent.left = new_node
    else:
        parent.right = new_node
    new_node.parent = parent
    if random.random() < 0.5:
        new_node.left = node
        new_node.right = Node(value={'role': 'D', 'triples': [], 'logical_rel': 'null'}, parent=new_node)
    else:
        new_node.right = node
        new_node.left = Node(value={'role': 'D', 'triples': [], 'logical_rel': 'null'}, parent=new_node)
    node.parent = new_node
    num_of_dummy_features += 1

def get_all_nodes(root):
    """Get all nodes from the root"""
    if root is None:
        return []
    return [root] + get_all_nodes(root.left) + get_all_nodes(root.right)

def get_a_child_tree(root, depth, p_prune):
    """Get a child tree from the root"""
    if root is None:
        return None, []
    if random.random() < 1 - p_prune - 0.05 * depth:
        return root, get_all_nodes(root)
    elif random.random() < 0.5:
        return get_a_child_tree(root.left, depth + 1, p_prune)
    else:
        return get_a_child_tree(root.right, depth + 1, p_prune)

def generate_subtree(original_tree_input, generation_params):
    """Generate a subtree from the original tree"""
    while True:
        original_tree = copy.deepcopy(original_tree_input)
        root, nodes = convert_full_bfs_tree_to_pointer_tree(original_tree)
        for node in nodes:
            if node.value['role'] == 'C':
                assert node.right is not None, "node.right is None"
                assert node.left is not None, "node.left is None"
        # 0. Get a child tree
        temp_root, temp_nodes = get_a_child_tree(root, 0, generation_params['p_prune'])
        if temp_root is not None:
            root, nodes = temp_root, temp_nodes
        for node in nodes:
            # 1. Determine whether to swap children
            if random.random() < generation_params['p_swap_children']:
                swap_children(node)
            # 2. Determine whether to deactivate a node
            if random.random() < generation_params['p_deactivate']:
                deactivate_node(node)
                continue # 如果节点被deactivate，则不进行后续操作
            # 3. Sample the condition set
            if node.value['role'] == 'C' and node.value['logical_rel'] != 'null':
                sample_condition_set(node, generation_params['p_keep_feature'])
            # 4. Add a dummy feature
            if node.value['role'] == 'C' and random.random() < generation_params['p_add_dummy']:
                add_dummy_feature(node)
            # 5. Add a dummy parent layer
            if random.random() < generation_params['p_add_dummy_parent_layer']:
                add_dummy_parent_layer(node)

        unfold_pointer_tree(root)
        bfs_tree = convert_pointer_tree_to_bfs_tree(root)
        if bfs_tree[0]['role'] == 'C':
            break
    return bfs_tree

def main():
    ##### 参数设置 #####
    dataset_path = "all_trees.json"
    num_sample_data = 7
    num_desired_subtrees = 10
    sample_params = {
        'depth_low': 3,
        'depth_high': 4,
        'K_low': 3,
        'K_high': 10,
        'condition_count_low':5,
        'condition_count_high': 10,
    }

    generation_params = {
        'p_swap_children': 0.05, # 交换子节点
        'p_prune': 0.05, # 剪枝
        'p_keep_feature': 0.8, # 保留特征
        'p_add_dummy': 0.05, # 添加虚拟特征
        'p_add_dummy_parent_layer': 0.05, # 添加虚拟父层
    }

    save_original_tree = True
    ##### 参数设置 #####

    # 1. Sample trees
    sampled_trees, disease_names = sample_trees(dataset_path, num_sample_data, sample_params)

    # 2. Generate subtrees from each tree
    timestamp = datetime.now().strftime("%m%d_%H%M%S") # 添加时间戳
    for sampled_tree, disease_name in zip(sampled_trees, disease_names):
        num_desired_subtrees = 10
        # Save the original and unfolded version of the original tree
        original_tree = sampled_tree["bfs_tree"]
        generated_subtrees = [original_tree]
        original_tree = copy.deepcopy(original_tree)
        root, nodes = convert_full_bfs_tree_to_pointer_tree(original_tree)
        unfold_pointer_tree(root)
        bfs_tree = convert_pointer_tree_to_bfs_tree(root)
        generated_subtrees.append(bfs_tree)

        # Generating subtrees
        for _ in range(num_desired_subtrees):
            new_subtree = generate_subtree(original_tree, generation_params)
            generated_subtrees.append(new_subtree)

        # 保存抽样的树和子树
        os.makedirs(f"data/{timestamp}/{disease_name}", exist_ok=True)
        subtrees_output_path = f"data/{timestamp}/{disease_name}/generated_subtrees.json"
        original_tree_output_path = f"data/{timestamp}/{disease_name}/sampled_tree.json"
        with open(subtrees_output_path, "w") as f:
            if save_original_tree:
                json.dump(generated_subtrees, f, indent=2) 
            else:
                json.dump(generated_subtrees[1:], f, indent=2) # 去掉原始树和展开的树
        with open(original_tree_output_path, "w") as f:
            json.dump(sampled_tree, f, indent=2)
        plot_tree(subtrees_output_path)

    # --- Output key file paths for downstream tasks ---
    print(f"Generated Subtrees JSON: {os.path.abspath(subtrees_output_path)}")
    print(f"Sampled Tree JSON: {os.path.abspath(original_tree_output_path)}")
    # Also print the directory containing these files
    print(f"Output Directory: {os.path.abspath(os.path.dirname(subtrees_output_path))}")

if __name__ == "__main__":
    main()