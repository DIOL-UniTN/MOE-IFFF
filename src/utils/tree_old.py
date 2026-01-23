import torch
import random
import math
from typing import Union

def tree_strtolist(s) -> list[list[int]]:
    nums = [int(c) for c in s]
    result = []
    i = 0
    level = 0
    while i < len(nums):
        count = 2 ** level
        result.append(nums[i:i+count])
        i += count
        level += 1
    return result

def fix_tree(encoded: str, depth:int) -> str:
    encoded, n = list(encoded), 2 ** (depth + 1) - 1
    for i in range(n):
        left, right = 2 * i + 1, 2 * i + 2
        if encoded[i] == '0' or encoded[i] == '2': # if leaf or none, cannot have children
            for child in (left, right):
                if child < n and encoded[child] != '0':
                    encoded[child] = '0'
        elif encoded[i] == '1':  # If  node 
            if i >= 2 ** depth - 1:  # Last depth can only be a leaf
                encoded[i] = '2'
            else: # Node must have children
                for child in (left, right):
                    if child < n and encoded[child] == '0': 
                        encoded[child] = '2'
    return ''.join(encoded)

def simplify_tree(tree: Union[str, list[int]]) -> str:
# def simplify_tree(tree: Union[str, list[int]], depth: int) -> str:
    # Simplify the tree by removing unnecessary layers
    tree = ''.join([str(c) for c in tree]) if type(tree) == list else tree
    depth = int(math.log2(len(tree)+1) - 1)
    for d in range(depth+1):
        first, last = 2 ** d - 1, 2 ** (d + 1) - 1
        if tree[first:last] == '0' * 2**d:
            return tree[:first]
    return tree

def gen_random_tree(depth: int, alpha: float = 0.1):
    n = 2 ** (depth + 1) - 1
    encoding = [str(random.choices([1, 2], weights=[1-alpha, alpha])[0])] # single layer FF with 10% probability
    remaining = n - 1 # Remaining nodes: random 0 or 1
    if remaining > 0:
        encoding += [str(random.randint(0, 2)) for _ in range(remaining)]
    return simplify_tree(fix_tree(''.join(encoding), depth))

def get_unbalanced_leaf_ids(arch_list: list[list[int]]) -> list[int]:
    leaf_ids, leaf_depths = [], []
    depth = len(arch_list) - 1
    for d, d_arch in enumerate(arch_list):
        platform = 2 ** d - 1
        d_leaf_ids = torch.where(torch.tensor(d_arch) == 2)[0] + platform
        for _ in range(depth - d): # Remaining depth 'till the leaves
            d_leaf_ids = 2 * (d_leaf_ids) + 1
        leaf_ids += d_leaf_ids.tolist()
        leaf_depths += [d] * d_arch.count(2)
    leaf_ids = [i - 2**(depth) + 1 for i in leaf_ids]
    return leaf_ids, leaf_depths
