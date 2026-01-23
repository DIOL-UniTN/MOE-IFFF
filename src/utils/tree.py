import torch
import math
import random
from typing import Optional

class Tree:
    def __init__(self, arch: Optional[list[int]] = None, max_depth: Optional[int] = None,
                 max_width: Optional[int] = None):
        self.arch = arch
        self.max_width = max_width
        self.max_depth = max_depth
        self.max_nodes = 2**(max_depth+1) - 1 if max_depth else None

    def __call__(self) -> list[int]:
        if self.arch == None:
            self.arch = self._gen_random_arch()
        arch = self._fix_tree(self.arch)
        self.arch = self._simplify_tree(arch)
        return self.arch

    def get_node(self, node_id: int) -> str:
        if node_id == -1:
            return "none"
        elif node_id == 0:
            return "router"
        elif node_id >= 1:
            return "leaf"
        else:
            assert False, f"Invalid node id: {node_id}"

    def get_node_id(self, node: str, width: Optional[int] = 1) -> int:
        if node == "none":
            return -1
        elif node == "router":
            return 0
        elif node == "leaf": # Put 
            return width
        else:
            assert False, f"Invalid node type: {node}"

    def _simplify_tree(self, arch) -> list[int]:
        depth = int(math.log2(len(arch)+1) - 1)
        for d in range(depth+1):
            first, last = 2 ** d - 1, 2 ** (d + 1) - 1
            if arch[first:last] == [self.get_node_id("none")] * 2**d:
                return arch[:first]
        return arch

    def _fix_tree(self, nodes: list[str]) -> list[int]:
        depth = int(math.log2(len(nodes)+1) - 1)
        for i, n in enumerate(nodes):
            left_c, right_c = (2 * i + 1, 2 * i + 2)
            children = [c for c in (left_c, right_c) if c < len(nodes)]
            for child in children:
                if (self.get_node(n) == "router" and
                    self.get_node(nodes[child]) == "none"):
                    nodes[child] = self.get_node_id("leaf", width=1) # if not given, make it a leaf with a width of only 1
                elif self.get_node(n) != "router": # if leaf or none, cannot have children
                    nodes[child] = self.get_node_id("none")
            # Fix final depth routers
            if (i >= 2**depth - 1 and 
                self.get_node(n) == "router"):
                nodes[i] = self.get_node_id("leaf", width=1)
        return nodes

    def _gen_random_balanced_arch(self) -> list[int]:
        if self.max_depth == None or self.max_width == None:
            assert False, "Max depth and Max width must be specified for random architecture generation"
        depth = random.randint(1, self.max_depth)
        n_nodes = 2**(depth+1) - 1
        nodes = [self.get_node_id("router")] * (2**depth - 1)
        nodes += [self.get_node_id("leaf")] * (2**depth)
        nodes += [self.get_node_id("none")] * (self.max_nodes - n_nodes)
        nodes = [n + random.randint(0, self.max_width-1) if n == 1 else n for n in nodes] # Random generation of leaf widths
        return nodes

    def _gen_random_arch(self) -> list[int]:
        if self.max_depth == None or self.max_width == None:
            assert False, "Max depth and Max width must be specified for random architecture generation"
        nodes = [self.get_node_id("router")] + [random.randint(-1, 1) 
                                                for _ in range(self.max_nodes - 1)] # Random generation of a node whether none, router or  leaf (width=1)
        nodes = [n + random.randint(0, self.max_width-1) if n == 1 else n for n in nodes] # Random generation of leaf widths
        return nodes

    def leaf_count(self, nodes) -> int:
        return sum([n > self.get_node_id("router") for n in nodes])

    def router_count(self, arch) -> int:
        return arch.count(self.get_node_id("router"))

    def get_leaf_widths(self) -> list[int]:
        return [int(n) for n in self.arch if n > 0] # Widths of the leaves

    def gen_tree_depths(self) -> list[list[int]]:
        result = []
        i, level = 0, 0
        while i < len(self.arch):
            count = 2 ** level
            result.append(self.arch[i:i+count])
            i += count
            level += 1
        return result

    def get_leaf_ids(self) -> tuple[list[int], list[int]]:
        arch_depth = self.gen_tree_depths()
        leaf_ids, leaf_depths = [], []
        depth = len(arch_depth) - 1
        for d, d_arch in enumerate(arch_depth):
            platform = 2 ** d - 1
            d_leaf_ids = torch.where(torch.tensor(d_arch) > 0)[0] + platform
            for _ in range(depth - d): # Remaining depth 'till the leaves
                d_leaf_ids = 2 * (d_leaf_ids) + 1
            leaf_ids += d_leaf_ids.tolist()
            leaf_depths += [d] * self.leaf_count(d_arch)
        leaf_ids = [i - 2**(depth) + 1 for i in leaf_ids]
        return leaf_ids, leaf_depths

    def get_router_ids(self) -> tuple[list[int], list[int]]:
        arch_depth = self.gen_tree_depths()
        router_ids, router_depths = [], []
        depth = len(arch_depth) - 1
        for d, d_arch in enumerate(arch_depth):
            platform = 2 ** d - 1
            d_router_ids = torch.where(torch.tensor(d_arch) == 0)[0] + platform
            for _ in range(depth - d): # Remaining depth 'till the leaves
                d_router_ids = 2 * (d_router_ids) + 1
            router_ids += d_router_ids.tolist()
            router_depths += [d] * self.router_count(d_arch)
        router_ids = [i - 2**(depth) + 1 for i in router_ids]
        return router_ids, router_depths
