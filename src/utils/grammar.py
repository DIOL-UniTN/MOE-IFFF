import random
from typing import Optional

class Node:
    def __init__(self, kind, left=None, right=None):
        self.kind = kind  # router or leaf
        self.left = left
        self.right = right

class DDTGrammar:
    def __init__(self, start: str, children: list[str], gene_bits: int, 
                 max_depth: int, max_leaf_width: int, depthwise: bool = False):
        self.depthwise = depthwise
        self.start = start
        self.children = children
        self.len_gene = 2**(max_depth+2) - 1
        self.gene_bits = gene_bits
        self.max_depth = max_depth
        self.high = 2**gene_bits - 1
        self.max_leaf_width = max_leaf_width
        self.leaf = [i+1 for i in range(max_leaf_width)]
        self.N = {
                "children": self.children,
                "leaf": self.leaf,
                }

    def is_phenotype_valid(self, phenotype: str) -> bool:
        for non_term in self.N.keys():
            if non_term in phenotype:
                return False
        return True

    def phenotype_to_tree(self, phenotype: str) -> list[int]:
        phenotype = phenotype.replace("router", "r")
        tree = self.parse_tree(phenotype)
        tree = self.to_array_by_levels(tree, max_depth=self.max_depth)
        tree[-1] = [1 if n == 0 else n for n in tree[-1]] # Punishing the router decisions
        arch = []
        for n in tree:
            arch.extend(n)
        return arch

    def gene_to_phenotype(self, genotype: list[int]) -> tuple[list[str], list[int]]:
        phenotype = self.start
        for gene in genotype:
            if "leaf" in phenotype:
                non_term_set = self.N["leaf"]
                choice_id = gene % len(non_term_set)
                choice = non_term_set[choice_id]
                phenotype = phenotype.replace(f"leaf", f"{choice}", 1)
            elif "children" in phenotype:
                non_term_set = self.N["children"]
                choice_id = gene % len(non_term_set)
                choice = non_term_set[choice_id]
                phenotype = phenotype.replace(f" children", f"({choice})", 1)
            else:
                break
        return phenotype

    def gene_to_phenotype_depthwise(self, genotype: list[int]) -> tuple[list[str], list[int]]:
        phenotype = self.start
        for gene in genotype:
            if "leaf" in phenotype:
                non_term_set = self.N["leaf"]
                choice_id = gene % len(non_term_set)
                choice = non_term_set[choice_id]
                phenotype = phenotype.replace(f"leaf", f"{choice}", 1)
            elif "children" in phenotype:
                non_term_set = self.N["children"]
                choice_id = gene % len(non_term_set)
                choice = non_term_set[choice_id].replace("children", "freeze")
                phenotype = phenotype.replace(f" children", f"({choice})", 1)
            elif "freeze" in phenotype:
                phenotype = phenotype.replace("freeze", "children") # refreeze children
                non_term_set = self.N["children"]
                choice_id = gene % len(non_term_set)
                choice = non_term_set[choice_id].replace("children", "freeze")
                phenotype = phenotype.replace(f" children", f"({choice})", 1)
            else:
                break
        return phenotype

    def gene_to_tree(self, genotype: list[str]) -> Optional[tuple[list[int], list[int]]]:
        phenotype = self.gene_to_phenotype_depthwise(genotype) if self.depthwise else self.gene_to_phenotype(genotype)
        phenotype = self.fix_unfinished_tree(phenotype)
        arch = self.phenotype_to_tree(phenotype)
        return arch

    def fix_unfinished_tree(self, phenotype: str) -> str:
        phenotype = phenotype.replace(f" freeze", f"(leaf,leaf)")
        phenotype = phenotype.replace(f" children", f"(leaf,leaf)")
        phenotype = phenotype.replace(f"leaf", f"1")
        return phenotype

    def get_config(self) -> dict:
        return {
                'start': self.start,
                'children': "|".join(self.children),
                'leaf': "|".join(map(str, self.leaf)),
                'len_gene': self.len_gene,
                'gene_bits': self.gene_bits,
                'max_leaf_width': self.max_leaf_width,
                }

    def parse_tree(self, s: str) -> Node:
        """Parses a string like 'r(r(l,l),l)' into an unbalanced tree."""
        import re
        tokens = re.findall(r'r|\d+|[()]|,', s.replace(" ", ""))
        pos = 0
        def parse():
            nonlocal pos
            if self.is_int(tokens[pos]):
                width = int(tokens[pos])
                pos += 1
                return Node(width)
            elif tokens[pos] == 'r':
                pos += 1
                assert tokens[pos] == '('
                pos += 1
                left = parse()
                assert tokens[pos] == ','
                pos += 1
                right = parse()
                assert tokens[pos] == ')'
                pos += 1
                return Node('r', left, right)
            else:
                raise ValueError("Unexpected token: " + tokens[pos])
        return parse()

    def is_int(self, s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    def to_array_by_levels(self, root: Optional[Node], max_depth: int) -> Optional[list[list[int]]]:
        """Convert tree to array of arrays by depth, fill gaps with -1."""
        if root is None:
            return None
        def height(node):
            if not node or type(node.kind) == int:
                return 0
            return 1 + max(height(node.left), height(node.right))

        h = height(root)
        tree = []
        queue = [root]

        for _ in range(h + 1):
            level = []
            new_queue = []
            for node in queue:
                if node is None:
                    level.append(-1)
                    new_queue.extend([None, None])
                else:
                    level.append(0 if node.kind == 'r' else node.kind)
                    if node.kind == 'r':
                        new_queue.append(node.left)
                        new_queue.append(node.right)
                    else:
                        new_queue.extend([None, None])
            tree.append(level)
            queue = new_queue
            if len(tree) == (max_depth + 1):
                return tree
        return tree
