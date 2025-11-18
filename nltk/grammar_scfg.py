# Natural Language Toolkit: Context Free Grammars
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Nicola Calzone <nicolacalzone14@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Data classes for representing Synchronous Context Free Grammars ("SCFG").  

An SCFG defines a translation between two ordinary Context-Free Grammars
(CFGs).  It generalises the single-language ``CFG`` implemented in
``grammar.py`` by combining three components:

1. Source CFG - the grammar for the *first* language.  
2. Target CFG - the grammar for the *second* language.  
3. Pairing relation - a set of synchronous productions that link a
   source rule to a corresponding target rule, together with an explicit
   alignment of their non-terminals.

Each synchronous production has the form:
    lhs -> (source_rhs, target_rhs)   [with index alignment]

where the index lists enforce a one-to-one correspondence between the
non-terminals on the two right-hand sides. 
These structures are historically known as Syntax-Directed Translation
Schemata (SDTS) in compiler theory and were later adopted in early
rule-based machine translation systems.  For a thorough introduction see:
    - David Chiang, “An Introduction to Synchronous Grammars”, 2006 
      link to paper: https://www3.nd.edu/~dchiang/papers/synchtut.pdf

To find out more about such structures, which were once common in compilers
under the name of Syntax-Directed Translation Schemata (SDTS), and then in
first experiments of Rule Based Machine Translation, you can look up at this
paper of David Chiang titled: "An Introduction to Synchronous Grammars", 2006,
at the following link: https://www3.nd.edu/~dchiang/papers/synchtut.pdf

#################################################################
# Synchronous Context-Free Grammars (SCFG)
#################################################################

The ``TreeSynCFG`` class encodes a synchronous context-free grammar (SCFG).  
An SCFG consists of a start symbol and a set of synchronous productions.  
The start symbol (by default the non-terminal ``S``) is the root of both the source-tree
and the target-tree that will be generated in parallel.

A synchronous production has the form  
* lhs - a ``Nonterminal`` that is expanded in *both* languages.  
* source_rhs - the right-hand side for the source (first) language.  
* target_rhs - the right-hand side for the target (second) language.  
* indexes - two parallel lists that explicitly align the non-terminals of the two RHSs
  (the same index appears on both sides for corresponding non-terminals).

Synchronous productions are implemented by the ``SynchronousProduction`` class.  
Each ``SynchronousProduction`` stores:

* ``_lhs`` - the left-hand side non-terminal.  
* ``_source_rhs`` / ``_target_rhs`` - tuples of symbols 
    (``Nonterminal`` or terminal strings).  
* ``_indexes_source`` / ``_indexes_target`` - integer indexes that must be identical for
  corresponding non-terminals.

The ``Nonterminal`` class is still used to distinguish node values from leaf values,
exactly as in ordinary CFGs.  Within a ``TreeSynCFG`` every node value is wrapped in a
``Nonterminal``; the generated trees, however, contain the plain symbol strings.

---

### Tree representation

During generation the class builds two parallel tree structures using the inner
``TreeNode`` class:

* ``_value`` - the symbol (terminal or non-terminal).  
* ``_index`` - the alignment index taken from the chosen production.  
* ``_children`` - ordered list of child ``TreeNode`` objects.  
* ``_linked_node`` - (optional) a direct reference to the corresponding node in the
  other tree (not used in the public API but kept for possible extensions).

After expansion the target tree is sorted by index (`sort_children`) so that the
leaves appear in the order defined by the grammar's alignment.

---

### Generation API

```python
scfg = TreeSynCFG.fromstring(grammar_string)      # parse textual SCFG
scfg.set_initial_depth(depth)                     # optional depth limit
src_tree, src_sent, tgt_tree, tgt_sent = scfg.produce(p_factor)



#################################################################
    # Very important note
#################################################################
    
   Limitation on Rule Complexity

   This implementation only supports synchronous productions with at most
   two non-terminals in each right-hand side (source and target).

   While the SCFG formalism permits arbitrary numbers of non-terminals per rule
   (e.g., ``S -> A{1} B{2} C{3} // C{3} B{2} A{1}``), such high-arity rules are
   not supported by the current parser and generator.

   Reason: Parsing and generation become computationally intractable beyond
   binary branching without significant algorithmic extensions (e.g., dynamic
   programming over hypergraphs). Since this module includes a working parser,
   we enforce this restriction to ensure correctness and efficiency.

   ---

   Allowed production types

   1. Binary rules - exactly two non-terminals (possibly with terminals):
      ```
      S -> A{1} B{2} // B{2} A{1}
      VP -> V{1} NP{2} // NP{2} V{1}
      ```

   2. Unary terminal rules - one terminal symbol:
      ```
      A -> walk // cammina
      B -> quickly // rapidamente
      ```

   ---

   Unsupported (will raise errors or fail silently)

   - Rules with three or more non-terminals:
     ```
     S -> A{1} B{2} C{3} // C{3} B{2} A{1}   not supported
     ```
   - Mixed high-arity rules, even if monotonic.

   Recommendation: Restructure your grammar using binarization techniques
   (introducing intermediate non-terminals) if you need to model complex
   alignments. Future versions may lift this restriction with advanced
   parsing strategies.

"""

import os
import sys
from nltk import Nonterminal
import random as rand
import re
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



#################################################################
# Grammars
#################################################################

class TreeSynCFG:

    class TreeNode:
        def __init__(self, value, index):
            self._value = value      # Symbol of the node (e.g., 'A', 'B', 'a')
            self._index = index      # Index of the node for the generation of the sentence
            self._children = []      # List of children of the node
            self._linked_node = None # Reference to a node in another tree

        # Allow constructing a node without explicitly providing an index
        # (convenience for various call-sites in generation code).
        def __init__(self, value, index=0):
            self._value = value
            self._index = index
            self._children = []
            self._linked_node = None

        def add_child(self, child):
            self._children.append(child)

        def set_children(self, children):
            self._children = children

        def get_children(self):
            return self._children

        def get_value(self):
            return self._value

        def get_index(self):
            return self._index

        def set_index(self, index):
            self._index = index

        def link_to(self, other_node):
            """Link this node to a node in another tree."""
            if isinstance(other_node, TreeSynCFG.TreeNode):
                self._linked_node = other_node
            else:
                raise ValueError("other_node must be an instance of TreeNode")

        def __repr__(self, level=0):
            ret = "\t" * level + repr(self._value) + " (Index: {})\n".format(self._index)
            for child in self._children:
                ret += child.__repr__(level + 1)
            #if self._linked_node:
            #    ret += "\t" * level + "Linked to: " + repr(self._linked_node._value) + "\n"
            return ret

        def sort_children(self):
            """Sort children at each level based on the index, with integers before 't'."""
            # Sort children: first by integer indices, then by 't' if present
            self._children.sort(key=lambda x: (x._index))

            # Recursively sort children for each child node
            for child in self._children:
                child.sort_children()

            return self


    #################################################################
    # Productions
    #################################################################

    class SynchronousProduction:
        """
        lhs -> (source_rhs, target_rhs)
        """

        def __init__(self, lhs, source_rhs, target_rhs, indexes_source, indexes_target):
            ## Given
            self._lhs = lhs
            self._source_rhs = source_rhs               # right-hand side for the source language - FIRST LANGUAGE
            self._target_rhs = target_rhs               # right-hand side for the target language - SECOND LANGUAGE
            self._indexes_source = indexes_source
            self._indexes_target = indexes_target

            ## Derived
            self._same_order()

        def lhs(self):
            """Return the left-hand side nonterminal."""
            return self._lhs

        def source_rhs(self):
            """Return the right-hand side for the source language."""
            return self._source_rhs

        def target_rhs(self):
            """Return the right-hand side for the target language."""
            return self._target_rhs

        def source_indexes(self):
            """Return the indexes of the source language."""
            return self._indexes_source

        def target_indexes(self):
            """Return the indexes of the target language."""
            return self._indexes_target

        def _same_order(self):
            """Return True if the source and target indexes are the same."""
            self._same_order = self._indexes_source == self._indexes_target

        def list_source_elements(self):
            return [ProductionElement(elem, idx) for elem, idx in zip(self._source_rhs, self._indexes_source)]

        def list_target_elements(self):
            return [ProductionElement(elem, idx) for elem, idx in zip(self._target_rhs, self._indexes_target)]

        def __repr__(self):
            """Return a string representation of the production."""
            return f"{self._lhs} -> {' '.join(map(str, self._source_rhs))} // {' '.join(map(str, self._target_rhs))}"


    def __init__(self, start, productions):
        self._start = start
        self._productions = productions
        self._translated_grammar = []
        self._initial_depth = None
        self._calculate_grammar_forms()

    def set_initial_depth(self, depth):
        self._initial_depth = depth

    def get_productions(self):
        return self._productions

    def get_start(self):
        return self._start

    def list_productions(self):
        list_elements = []
        for prod in self._productions:
            list_elements.append((prod.list_source_elements(), prod.list_target_elements()))
        return list_elements

    def _calculate_grammar_forms(self):
        """
        Pre-calculate of which form(s) the grammar is.
        """
        prods = self._productions
        self._min_len_source = min(len(sync_prod.source_rhs()) for sync_prod in prods)
        self._max_len_source = max(len(sync_prod.source_rhs()) for sync_prod in prods)
        self._min_len_target = min(len(sync_prod.target_rhs()) for sync_prod in prods)
        self._max_len_target = max(len(sync_prod.target_rhs()) for sync_prod in prods)

    def is_binarised(self):
        """
        Return True if all productions are at most binary.
        Note that there can still be empty and unary productions.
        """
        return self._max_len_source <= 2 and self._max_len_target <= 2

    @staticmethod
    def error_checker(source_rhs, target_rhs, source_indexes, target_indexes):
        """
        Check for errors in the productions.
        """

        if len(source_rhs) != len(target_rhs):
            raise ValueError(f"Source and target right-hand sides must have the same length. Found: {len(source_rhs)} and {len(target_rhs)}")

        if len(source_indexes) != len(target_indexes):
            raise ValueError(f"Source and target indexes must have the same length. Found: {len(source_indexes)} and {len(target_indexes)}")

        if len(source_rhs) != len(source_indexes):
            raise ValueError(f"Source right-hand side and indexes must have the same length. Found: {len(source_rhs)} and {len(source_indexes)}")

        if len(target_rhs) != len(target_indexes):
            raise ValueError(f"Target right-hand side and indexes must have the same length. Found: {len(target_rhs)} and {len(target_indexes)}")

        # Check if # of nonterminals match
        source_counter = Counter([elem.symbol() for elem in source_rhs if isinstance(elem, Nonterminal)])
        target_counter = Counter([elem.symbol() for elem in target_rhs if isinstance(elem, Nonterminal)])
        if source_counter != target_counter:
            raise ValueError("Nonterminal counts do not match between source and target RHS")

        # Check if indexes match for each nonterminal
        source_index_map = {f"{elem.symbol()}{idx}": idx for elem, idx in zip(source_rhs, source_indexes) if isinstance(elem, Nonterminal)}
        target_index_map = {f"{elem.symbol()}{idx}": idx for elem, idx in zip(target_rhs, target_indexes) if isinstance(elem, Nonterminal)}
        if source_index_map != target_index_map:
            raise ValueError("Nonterminal indexes do not match between source and target RHS")

    @classmethod
    def fromstring(cls, grammar_str: str):
        """
        Parses a grammar string into a list of productions.

        Args:
            grammar_str (str): The grammar string.

        Returns:
            list: A list of productions.
        """
        productions = []
        for line in grammar_str.strip().splitlines():
            if line:
                lhs, source_rhs, target_rhs = parse_grammar_line(line)

                # Parse source and target RHS into elements
                source_elements = re.findall(r'(\w+)(?:\{(\d+)?\})?', source_rhs)
                target_elements = re.findall(r'(\w+)(?:\{(\d+)?\})?', target_rhs)

                # Process source and target elements
                source_rhs_clean, source_indexes = process_elements(source_elements)
                target_rhs_clean, target_indexes = process_elements(target_elements)

                TreeSynCFG.error_checker(source_rhs_clean, target_rhs_clean, source_indexes, target_indexes)

                lhs = Nonterminal(lhs)
                prod = TreeSynCFG.SynchronousProduction(lhs, source_rhs_clean, target_rhs_clean, source_indexes, target_indexes)
                productions.append(prod)

        return TreeSynCFG(Nonterminal('S'), productions)

    def _choose_production(self, symbol: str, p_factor: float, depth: int):
        """
        Choose a production for the given nonterminal symbol, favoring terminal productions as depth decreases.
        Args:
            symbol:     str
                The symbol for which to choose a production

            p_factor:   float
                Probability factor to choose terminal productions

            depth:      int
                Current depth of the tree

        """
        # applicable_productions: Find applicable productions for the given symbol
        applicable_productions = [prod for prod in self._productions if prod.lhs() == symbol]

        # terminal_productions: Find terminal productions where RHS contains only terminals
        terminal_productions = [prod for prod in applicable_productions
                                if all(not isinstance(sym, Nonterminal) for sym in prod.source_rhs())]

        # expandable_productions: Find productions that are not classified as terminal
        expandable_productions = [prod for prod in applicable_productions if prod not in terminal_productions]

        if terminal_productions and rand.random() < p_factor:
            chosen_production = rand.choice(terminal_productions)
            return chosen_production

        if expandable_productions:
            chosen_production = rand.choice(expandable_productions)
            return chosen_production

        return None

    def _generate_trees(self, p_factor: float, depth: int, source_symbol="S", target_symbol="S", debug=False):
        """Generate trees for both source and target synchronously."""

        applicable_productions = [prod for prod in self._productions if prod.lhs() == source_symbol]
        terminal_productions = [ prod for prod in applicable_productions
                                if all(not isinstance(sym, Nonterminal) for sym in prod.source_rhs()) ]

        if depth <= 0 and source_symbol in terminal_productions:
            return TreeSynCFG.TreeNode(source_symbol), TreeSynCFG.TreeNode(target_symbol)

        source_node = TreeSynCFG.TreeNode(source_symbol)
        target_node = TreeSynCFG.TreeNode(target_symbol)

        # Choose a production for the given symbol
        adjusted_p_factor = p_factor + (1.0 - p_factor) * (1 - (depth / self._initial_depth))
        chosen_production = self._choose_production(source_symbol, adjusted_p_factor, depth)

        if not chosen_production:   # No productions available
            return TreeSynCFG.TreeNode(source_symbol), TreeSynCFG.TreeNode(target_symbol)

        source_rhs = chosen_production.source_rhs()
        target_rhs = chosen_production.target_rhs()
        source_indexes = chosen_production.source_indexes()
        target_indexes = chosen_production.target_indexes()

        for i, source_sym in enumerate(source_rhs):
            source_child, target_child = self._generate_trees(p_factor, depth - 1,
                                                            source_sym, target_rhs[i],
                                                            debug)

            source_node.add_child(source_child)
            target_node.add_child(target_child)

            #source_node.link_to(target_node)
            #target_node.link_to(source_node)

            if i < len(source_indexes) and i < len(target_indexes):
                source_child.set_index(source_indexes[i])
                target_child.set_index(target_indexes[i])

        return source_node, target_node

    def _generate_sentence(self, node, debug=False):
        """Convert the tree into a sentence."""
        if node is None or not node.get_children():
            return node.get_value() if node else ""  # Nodo terminale: restituisce il valore

        sentence = []
        for child in node.get_children():
            child_sentence = self._generate_sentence(child, debug)
            if not isinstance(child_sentence, Nonterminal):
                sentence.append(str(child_sentence))

        return " ".join(sentence)

    def produce(self, p_factor: float, debug=False): # depth: int
        """Generate trees and sentences for both source and target."""

        depth = self._initial_depth
        source_tree, target_tree = self._generate_trees(p_factor, depth, self._start, self._start, debug)
        target_tree_reordered = target_tree.sort_children()

        source_sentence = self._generate_sentence(source_tree, debug)
        target_sentence = self._generate_sentence(target_tree_reordered, debug)

        return source_tree, source_sentence, target_tree_reordered, target_sentence


    @classmethod
    def translate_grammar_for_parser(cls, grammar_str: str):
        """
        Transforms a grammar string into the desired structure.

        Args:
            grammar_str (str): The grammar string.

        Returns:
            list: A list of tuples representing the grammar rules.
        """
        g5_parser = []
        for line in grammar_str.strip().splitlines():
            if line:
                lhs, source_rhs, target_rhs = parse_grammar_line(line)

                # Parse source and target RHS into ProductionElement instances
                source_elements = [
                    ProductionElement(symbol, int(index) if index else 0)
                    for symbol, index in re.findall(r'(\w+)(?:\{(\d+)?\})?', source_rhs)
                ]
                target_elements = [
                    ProductionElement(symbol, int(index) if index else 0)
                    for symbol, index in re.findall(r'(\w+)(?:\{(\d+)?\})?', target_rhs)
                ]

                # Add to g5_parser
                g5_parser.append((lhs, source_elements, target_elements))

        return g5_parser




class ProductionElement:
    """
    Represents an element in a production rule.
    """
    def __init__(self, symbol, index):
        ## Given
        self._symbol = symbol
        self._index = index

        ## Derived
        if isinstance(symbol, Nonterminal):
            self._isnonterminal = True
        else:
            self._isnonterminal = symbol.isupper()

    def symbol(self):
        return self._symbol

    def index(self):
        return self._index

    def isnonterminal(self):
        return self._isnonterminal

    def __repr__(self):
        return f"{self._symbol}, {self._index}, {self._isnonterminal}"

def parse_grammar_line(line: str):
    """
    Parses a single line of the grammar string into its components.

    Args:
        line (str): A line from the grammar string.

    Returns:
        tuple: (lhs, source_rhs, target_rhs)
    """
    if '//' not in line or "->" not in line:
        raise ValueError(f"Unable to parse line: {line}. Expected '//' and '->'.")

    source_rule, target_rhs = line.split('//')
    lhs, source_rhs = source_rule.split('->')
    lhs = lhs.strip()
    source_rhs = source_rhs.strip()
    target_rhs = target_rhs.strip()

    return lhs, source_rhs, target_rhs


def process_elements(elements):
    cleaned_elements = []
    indexes = []
    i = 0
    for elem, idx in elements:
        cleaned_elements.append(Nonterminal(elem) if elem.isupper() else elem)
        indexes.append(int(idx) if idx else i)
        i += 2 if elem.isupper() else 1
    return cleaned_elements, indexes


if __name__ == "__main__":
    """
    Quick test: build a small SCFG, generate trees and sentences.
    This is a simple smoke-test for the generation API. 
    """

    sample_grammar = """
    S  -> NP{1} VP{2}                // NP{1} VP{2}
    NP -> A{1} N{2}                  // A{1} N{2}
    NP -> N{1} A{2}                  // A{2} N{1} 
    VP -> V{1} NP{2}                 // NP{2} V{1}
    VP -> V{1} ADV{2} NP{3}          // ADV{2} NP{3} V{1}

    A   -> the       // il
    A   -> a         // un

    N   -> dog       // cane
    N   -> cat       // gatto
    N   -> john      // giovanni
    N   -> mary      // maria

    V   -> sees      // vede
    V   -> loves     // ama
    V   -> runs      // corre

    ADV -> quickly  // rapidamente
    ADV -> slowly   // lentamente
    """

    scfg = TreeSynCFG.fromstring(sample_grammar)
    scfg.set_initial_depth(4)

    print("Testing TreeSynCFG generation (5 samples):")
    for idx in range(5):
        src_tree, src_sent, tgt_tree, tgt_sent = scfg.produce(p_factor=0.7)
        print(f"\nSample #{idx + 1}")
        print("Source tree:")
        print(src_tree)
        print("Source sentence:", src_sent)
        print("Target tree:")
        print(tgt_tree)
        print("Target sentence:", tgt_sent)
        print("-" * 60)
