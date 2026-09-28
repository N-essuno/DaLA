"""Indexed acceleration of the existing trigram/SequenceMatcher deduplication.

No similarity threshold is relaxed. Rare-gram candidate retrieval uses a lower
bound on shared trigrams implied by >=90% matched tokens, then verifies the
original SequenceMatcher predicate. Tests compare against the original scan.
"""
from collections import Counter
from difflib import SequenceMatcher
from .curation import normalized_tokens


def required_shared_trigrams(n, m):
    matches = (9 * (n + m) + 19) // 20  # ceil(.9 * (n+m) / 2)
    unmatched = n + m - 2 * matches
    # Each additional matching block consumes at least one unmatched token.
    return max(1, matches - 2 * (unmatched + 1))


class IndexedNearDuplicates:
    def __init__(self):
        self.tokens = []
        self.grams = []
        self.index = {}
        self.exact = set()

    def add(self, text):
        tokens = normalized_tokens(text)
        if tokens in self.exact:
            return False
        grams = Counter(zip(tokens, tokens[1:], tokens[2:]))
        n = len(tokens)
        lengths = range((9*n+10)//11, (11*n)//9+1)
        # Omitting fewer occurrences than every qualifying pair must share
        # guarantees at least one remaining retrieval gram for such a pair.
        budget = min((required_shared_trigrams(n,m) for m in lengths), default=1)-1
        retrieve = set(grams)
        for gram in sorted(grams, key=lambda g: sum(len(v) for v in self.index.get(g,{}).values()), reverse=True):
            if grams[gram] <= budget:
                budget -= grams[gram]; retrieve.remove(gram)
        neighbours = set()
        for gram in retrieve:
            postings = self.index.get(gram,{})
            for length in lengths:
                neighbours.update(postings.get(length,()))
        for i in neighbours:
            other = self.tokens[i]
            if 2*min(n,len(other))/(n+len(other)) < .9:
                continue
            if sum((grams & self.grams[i]).values()) < required_shared_trigrams(n,len(other)):
                continue
            if SequenceMatcher(None,tokens,other,autojunk=False).ratio() >= .9:
                return False
        i = len(self.tokens)
        self.tokens.append(tokens);self.grams.append(grams);self.exact.add(tokens)
        for gram in grams:
            self.index.setdefault(gram,{}).setdefault(n,[]).append(i)
        return True
