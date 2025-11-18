# Natural Language Toolkit: Context Free Grammars
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Nicola Calzone <nicolacalzone14@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


"""
.. module:: parser_scfg
   :synopsis: Bottom-up chart parser for Synchronous Context-Free Grammars (SCFG)

========================================
Synchronous CFG Parser (`parser_scfg.py`)
========================================

This module implements a bottom-up chart parser for Synchronous Context-Free Grammars (SCFGs) 
with explicit non-terminal alignment via index annotations (e.g., ``A{1}``).

It works in tandem with :class:`TreeSynCFG` from :mod:`grammar_scfg`, which generates and 
represents SCFGs. The parser supports only binary and unary productions — a deliberate 
restriction to ensure efficient and correct parsing using a dynamic programming table.

---

Key Features
------------

- Synchronous span alignment: Parses two strings ``w1`` and ``w2`` simultaneously, respecting 
    the index-based correspondence defined in the grammar.
- Chart parsing (CKY-style): Uses a 4D table indexed by spans in both strings: 
    ``table[i][j][i'][j']``.
- Early failure detection: Rejects input if:
  - Strings have different lengths.
  - Unknown terminals appear.
  - Grammar contains unsupported rule arity.
- Memoization: Avoids re-proving failed or active items using ``active_spans`` and 
    ``non_working_spans``.

---

Limitations
-----------

.. important::

   Only binary and unary rules are supported.

   The parser does not support productions with more than two non-terminals (e.g., ``S -> A{1} B{2} C{3}``). Such grammars are theoretically valid but lead to intractable parsing complexity without hypergraph algorithms.

   Allowed rule types:

   1. Binary: ``A -> B{1} C{2} // C{2} B{1}``
   2. Unary terminal: ``A -> walk // cammina``
   3. Unary non-terminal: ``S -> NP{1} // NP{1}`` (for structure copying)

   Use grammar binarization if you need to model higher-arity alignments.

---

Classes
-------

.. autoclass:: Item
   :members:
   :undoc-members:
   :show-inheritance:

   A lightweight dataclass representing a parsing item (hypothesis) over synchronized spans.

.. autoclass:: SynchronousCFGParser
   :members: parse, can_prove_item
   :special-members: __init__

   Main parser class.

---

Example Usage
-------------

.. code-block:: python

   from grammar_scfg import TreeSynCFG
   from sync_parser import SynchronousCFGParser

   # Define a simple SCFG
   grammar_str = '''
   S -> NP{1} VP{2} // VP{2} NP{1}
   NP -> John // Giovanni
   VP -> walks // cammina
   '''

   # Build grammar
   scfg = TreeSynCFG.fromstring(grammar_str)
   rules = TreeSynCFG.translate_grammar_for_parser(grammar_str)

   # Parse
   parser = SynchronousCFGParser(rules)
   success = parser.parse("John walks", "Giovanni cammina")

   print("Parsed successfully:", success)  # True

---

See Also
--------

- :file:`grammar_scfg.py` - SCFG representation and sentence generation.
- David Chiang, “An Introduction to Synchronous Grammars”, 2006  
  https://www3.nd.edu/~dchiang/papers/synchtut.pdf

"""


from __future__ import annotations
from typing import List, Set, Tuple, Dict, DefaultDict
from collections import defaultdict

class SynchronousCFGParser:
    """
    CKY parser for **binary/unary** Synchronous Context-Free Grammars.

    Parameters
    ----------
    scfg : TreeSynCFG
        The synchronous grammar to parse with. Must contain only binary or unary
        productions.

    Raises
    ------
    ValueError
        If the grammar contains productions with >2 non-terminals.
    """

    def __init__(self, scfg): #: TreeSynCFG
        self.scfg = scfg
        self.start = str(scfg.get_start())

        # Validate binarity
        for prod in scfg.get_productions():
            src_len = sum(1 for sym in prod.source_rhs() if isinstance(sym, type(scfg.get_start())))
            tgt_len = sum(1 for sym in prod.target_rhs() if isinstance(sym, type(scfg.get_start())))
            if src_len > 2 or tgt_len > 2:
                raise ValueError(
                    f"Unsupported production (arity > 2): {prod}\n"
                    "Only binary/unary rules are allowed."
                )

        self._index_rules()
        self._index_terminals()

    def _index_terminals(self) -> None:
        """Pre-index terminal rules: (src_term, tgt_term) → LHS"""
        self.terminal_rules: Dict[Tuple[str, str], Set[str]] = {}
        for prod in self.scfg.get_productions():
            # use ProductionElement wrappers which expose .symbol(), .index(), .isnonterminal()
            src_elems = prod.list_source_elements()
            tgt_elems = prod.list_target_elements()
            if len(src_elems) == 1 and len(tgt_elems) == 1:
                s_elem = src_elems[0]
                t_elem = tgt_elems[0]
                if not s_elem.isnonterminal() and not t_elem.isnonterminal():
                    key = (str(s_elem.symbol()), str(t_elem.symbol()))
                    if key not in self.terminal_rules:
                        self.terminal_rules[key] = set()
                    self.terminal_rules[key].add(str(prod.lhs()))

    def _index_rules(self) -> None:
        """Index binary rules by LHS and child symbols + indices"""
        self.binary_rules: DefaultDict[str, List[tuple]] = defaultdict(list)
        # Format per entry: (B_sym, C_sym, B2_pos, C2_pos)
        for prod in self.scfg.get_productions():
            src_elems = prod.list_source_elements()
            tgt_elems = prod.list_target_elements()
            if len(src_elems) != 2 or len(tgt_elems) != 2:
                continue  # skip unary or terminal productions

            B1, C1 = src_elems
            # ensure source children are nonterminals
            if not (B1.isnonterminal() and C1.isnonterminal()):
                continue

            # Find positions in target (match both symbol and index)
            B2_pos = next((i for i, e in enumerate(tgt_elems)
                           if str(e.symbol()) == str(B1.symbol()) and str(e.index()) == str(B1.index())), None)
            C2_pos = next((i for i, e in enumerate(tgt_elems)
                           if str(e.symbol()) == str(C1.symbol()) and str(e.index()) == str(C1.index())), None)
            if B2_pos is None or C2_pos is None:
                continue

            self.binary_rules[str(prod.lhs())].append((
                str(B1.symbol()), str(C1.symbol()),
                int(B2_pos), int(C2_pos)
            ))

    def parse(self, w1: str, w2: str, debug: bool = False):
        """
        Parse a pair of strings synchronously.

        Parameters
        ----------
        w1 : str
            Source string.
        w2 : str
            Target string.
        debug : bool, optional
            If ``True``, returns non-empty table cells for debugging.

        Returns
        -------
        Tuple[bool, int]
            ``(success, operations)`` — whether ``S`` spans both full strings,
            and the number of rule applications attempted.

        Raises
        ------
        ValueError
            If inputs have different lengths or contain unknown terminals.
        """
        if not w1 or not w2:
            raise ValueError("Input strings cannot be empty")
        if len(w1) != len(w2):
            raise ValueError(f"Length mismatch: {len(w1)} != {len(w2)}")

        n = len(w1)
        ops = 0

        # 4D table: table[i][j][ip][jp] = set of non-terminals
        table: List[List[List[List[Set[str]]]]] = [
            [[[set() for _ in range(n + 1)] for _ in range(n + 1)]
             for _ in range(n + 1)] for _ in range(n + 1)
        ]

        # === TERMINAL RULES (length 1) ===
        for i in range(n):
            for ip in range(n):
                ops += 1
                t1, t2 = w1[i], w2[ip]
                lhs_set = self.terminal_rules.get((t1, t2))
                if lhs_set:
                    for lhs in lhs_set:
                        table[i][i + 1][ip][ip + 1].add(lhs)

        # === BINARY RULES (length > 1) ===
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for ip in range(n - length + 1):
                    jp = ip + length
                    for lhs, rules in self.binary_rules.items():
                        for B_sym, C_sym, B2_pos, C2_pos in rules:
                            ops += 1
                            for k in range(i + 1, j):
                                for kp in range(ip + 1, jp):
                                    # Source: B over [i,k], C over [k,j]
                                    if B_sym in table[i][k][ip if B2_pos == 0 else kp][kp if B2_pos == 0 else jp]:
                                        if C_sym in table[k][j][ip if C2_pos == 0 else kp][kp if C2_pos == 0 else jp]:
                                            table[i][j][ip][jp].add(lhs)

        success = self.start in table[0][n][0][n]

        if debug:
            # collect non-empty cells for debugging
            non_empty = []
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    for ip in range(n + 1):
                        for jp in range(ip + 1, n + 1):
                            if table[i][j][ip][jp]:
                                non_empty.append(((i, j, ip, jp), set(table[i][j][ip][jp])))
            return success, ops, non_empty

        return success, ops



if __name__ == "__main__":
    
    g_test = """
        S -> A{1} B{2} // B{2} A{1}
        A -> A{1} B{2} // A{1} B{2}
        A -> C{1} F{2} // C{1} F{2}
        B -> B{1} F{2} // F{2} B{1}
        B -> D{1} A{2} // D{1} A{2}
        C -> C{1} D{2} // D{2} C{1}
        C -> F{1} B{2} // B{2} F{1}
        D -> F{1} A{2} // F{1} A{2}
        D -> D{1} C{2} // D{1} C{2}
        F -> D{1} B{2} // B{2} D{1}
        F -> F{1} C{2} // C{2} F{1}
        A -> a // e
        A -> b // f
        A -> a // g
        B -> b // f
        B -> a // e
        C -> c // g
        C -> d // h
        D -> d // h
        D -> c // g
        F -> c // g
        F -> a // e
        """
    
    from .grammar_scfg import TreeSynCFG

    """ TreeSynCFG and SynchronousCFGParser usage example and test cases """
    scfg = TreeSynCFG.fromstring(g_test)    
    parser = SynchronousCFGParser(scfg)

    # Positive and negative test pairs
    positive_tests = [
        ("abcaadacdbcbbd", "hhgfgffeheefeg"),
        ("dcdcdccccaabdb", "fhhgggghgghfee"),
        ("aadccbabcacaca", "gggeeehfggfgeg"),
        ("ddccaaccacdacb", "fghheggeeggghe"),
        ("caccbddcacdcca", "ggghegegfgehhg"),
        ("bccdcabcbaaddb", "hfhgfeefgghfeg"),
        ("acbdaacacaddab", "ffgegeghhgehgg"),
        ("acacbcaaaadbab", "fggfggeeeehfee"),
        ("aacbaacacdcaaa", "hgegeegegfeege"),
        ("cddcdaaddaddbb", "hffheehehhhghg"),
    ]

    negative_tests = [
        ("baacbddddbdbdd", "egfefeehhhfgef"),
        ("cdbacdaadcbcbd", "geehfheegfehhf"),
        ("adadbbcdcdddab", "fheggefheehfee"),
        ("aadccacddcbadd", "ghfhfeggeehgeg"),
        ("cbabcacaadccac", "gfffgfhfggfggg"),
        ("baccbdadadcadc", "ghhgeehegghfhh"),
        ("adaadbccbbdaab", "egehfffhegfeef"),
        ("acdadccdbddbab", "fhhhggffhheggf"),
        ("bbddbadcbbacbc", "gggehghheegeeg"),
        ("adababbbbdbabd", "fghfgfefgfffee"),
    ]

    pos_correct = 0
    neg_correct = 0
    pos_failed = []
    neg_false_positives = []

    import time
    start = time.time()

    for s, t in positive_tests:
        success, _ops = parser.parse(s, t)
        if success:
            pos_correct += 1
        else:
            pos_failed.append((s, t))

    for s, t in negative_tests:
        success, _ops = parser.parse(s, t)
        if not success:
            neg_correct += 1
        else:
            neg_false_positives.append((s, t))

    end = time.time()

    total_pos = len(positive_tests)
    total_neg = len(negative_tests)

    print(f"Total parsing time: {end - start:.2f} seconds\n")
    print(f"Positive correct: {pos_correct}/{total_pos}  ({pos_correct/total_pos*100:.2f}%)")
    print(f"Negative correct: {neg_correct}/{total_neg}  ({neg_correct/total_neg*100:.2f}%)")
    
    if pos_failed:
        print("\nFailed positive examples:")
        for s,t in pos_failed:
            print(f"  src: {s}  tgt: {t}")
    if neg_false_positives:
        print("\nFalse positive (negative accepted) examples:")
        for s,t in neg_false_positives:
            print(f"  src: {s}  tgt: {t}")
            