from typing import *

import bisect
from colors import green
from functools import reduce
from hiertype.data.alphabet import Alphabet


class Hierarchy:
    """
    Stores a hierarchical type ontology for hierarchical classification.
    """
    OTHER = "<OTHER>"

    def __init__(self,
                 alphabet: Alphabet,
                 num_level: int,
                 level: List[range],
                 parent: List[int],
                 children: List[range],
                 dummies: Set[int],
                 with_other: bool,
                 delimiter: str = "/",
                 ):
        self.num_level = num_level
        self._alphabet = alphabet
        self._level = level
        self._parent = parent
        self._children = children
        self._dummies = dummies
        self.delimiter = delimiter
        self.with_other = with_other

    def type_str(self, i: int) -> str:
        """
        Returns the name of a type indexed by the specified id.
        :param i: ID
        :return: Type name
        """
        return self._alphabet.idx_to_sym[i]

    def index(self, t: str) -> int:
        """
        Returns the index of the specific type. If t is OOV, return 0.
        :param t: Type name
        :return: Index to that type
        """
        return self._alphabet.sym_to_idx.get(t) or 0

    def index_of_nodes_on_path(self, t: str) -> List[int]:
        """
        Returns the path of types from Any (the supertype of all types) to the given type.
        :param t: Type name
        :return: List of node indices, starting from 0 (Any)
        """
        arcs = t.split(self.delimiter)  # ["", a, b]
        prefixes = [self.delimiter.join(arcs[:i + 1]) for i in range(len(arcs))]
        nodes = []
        for prefix in prefixes:
            node = self._alphabet.sym_to_idx.get(prefix)
            if node is not None:
                nodes.append(node)

        if self.with_other:
            last_node = nodes[-1]
            if len(self.children(last_node)) != 0:  # not leaf
                other_node = self.other_child(last_node)
                if other_node is not None:
                    nodes.append(other_node)  # add <OTHER>
        return nodes

    def other_child(self, x: int) -> Optional[int]:
        """
        Given type t indexed by x, return the index of "t/<OTHER>".
        :param x: Index of parent node
        :return: Index of OTHER child node (None if no child exists)
        """
        if self.with_other:
            path = f"{self.type_str(x)}{self.delimiter}{Hierarchy.OTHER}"
            return self._alphabet.sym_to_idx.get(path)  # could be None if no other found
        else:
            return None

    def parent(self, x: int) -> int:
        """Returns the set of parent nodes of this node."""
        return self._parent[x]

    def path_to_root(self, x: int) -> List[int]:
        """Returns the path from the current node to the root."""
        path = [x]
        p = x
        while p != 0:
            p = self._parent[p]
            path.append(p)
        return path

    def children(self, x: int) -> range:
        """Returns the span of children nodes of this node."""
        c = self._children[x]
        return c if c is not None else range(0, 0)  # else branch: empty range

    def is_dummy(self, x: int) -> bool:
        return x in self._dummies

    def sibling(self, x: int) -> Set[int]:
        if x == 0:  # root
            return {0}
        else:
            return set(self._children[self._parent[x]])

    def level(self, x: int) -> int:
        for i in range(self.num_level):
            if self._level[i].start <= x < self._level[i].stop:
                return i

    def level_range(self, i: int) -> range:
        """Returns all nodes at level i."""
        return self._level[i]

    def size(self) -> int:
        return self._alphabet.size()

    def __str__(self):
        """
        Prints the hierarchy in a terminal-friendly way.
        """
        import io
        buf = io.StringIO()
        stack = [(0, "   ")]

        def next_prefix(prefix: str, last: bool) -> str:
            if prefix.endswith("├─"):
                return f"{prefix[:-3]} │ {' └─' if last else ' ├─'}"
            else:
                return f"{prefix[:-3]}   {' └─' if last else ' ├─'}"

        while len(stack) > 0:
            c, prefix = stack.pop()
            l = self.level(c)
            p = self.parent(c)
            print(f"{prefix} {green(str(c))} {self.type_str(c)}", file=buf)
            cc = self.children(c)
            if len(cc) != 0:
                cc = list(reversed(cc))
                stack.append((cc[0], next_prefix(prefix, True)))
                for d in cc[1:]:
                    stack.append((d, next_prefix(prefix, False)))

        return buf.getvalue()

    @classmethod
    def from_tree_file(cls, filename: str, with_other: bool = False, delimiter: str = '/'):

        def parent_type(t: str):
            p = delimiter.join(t.split(delimiter)[:-1])
            if all(c == delimiter for c in p):  # "//", "/" => ""
                p = ""
            return p

        types_per_level: List[Set[str]] = []
        with open(filename, mode='r') as file:
            for line in file:
                arcs = line.strip().split(delimiter)  # ["", "a", "b"]
                for i in range(1, len(arcs) + 1):
                    prefix = delimiter.join(arcs[:i])
                    if prefix.endswith(delimiter):
                        continue  # skip types like "/" or "//" that is the apparent parent of "//person"
                    if len(types_per_level) < i:
                        types_per_level.append(set())
                    types_per_level[i - 1].add(prefix)

                    if with_other and i < len(arcs):  # non-leaf
                        while len(types_per_level) <= i:
                            types_per_level.append(set())
                        types_per_level[i].add(f"{prefix}{delimiter}{Hierarchy.OTHER}")

        num_level = len(types_per_level)
        alphabet: Alphabet = Alphabet.with_special_symbols([])
        num_all_types = sum(len(types) for types in types_per_level)
        parent: List[int] = [0] * num_all_types
        children: List[range] = [None] * num_all_types

        level_start = 0
        level = []
        for l in range(num_level):
            types = sorted(types_per_level[l])
            level_size = len(types)
            level.append(range(level_start, level_start + level_size))
            level_start += level_size

            for k in range(level_size):
                i = alphabet.size()
                t = types[k]
                alphabet.index(t)
                if i != 0:  # not root
                    p = alphabet.sym_to_idx[parent_type(t)]
                    parent[i] = p
                    if children[p] is None:
                        children[p] = range(i, i + 1)
                    else:
                        r = children[p]
                        children[p] = range(min(i, r.start), max(i + 1, r.stop))

        dummies: Set[int] = {0}
        for i, t in enumerate(alphabet.idx_to_sym):
            if t.endswith(Hierarchy.OTHER):
                dummies.add(i)

        return cls(alphabet, num_level, level, parent, children, dummies, with_other, delimiter)
