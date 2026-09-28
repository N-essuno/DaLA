"""Conservative, single-span English corruptions over gold UD annotations.

No parser/model downloads, spelling noise, or arbitrary deletion fallback.
See docs/english.md for scope, exclusions, and validation requirements.
"""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Edit:
    corruption_type: str
    token_id: str
    start: int
    end: int
    original: str
    replacement: str

    def apply(self, text):
        if text[self.start:self.end] != self.original:
            raise ValueError("Edit does not match source text")
        return text[:self.start] + self.replacement + text[self.end:]


# Deliberately finite lexicons: avoid guessing inflection or dialectal variants.
BASE_FORMS = {
    "go": ("goes", "went"), "see": ("sees", "saw"),
    "take": ("takes", "took"), "make": ("makes", "made"),
    "give": ("gives", "gave"), "write": ("writes", "wrote"),
    "eat": ("eats", "ate"), "know": ("knows", "knew"),
    "find": ("finds", "found"), "leave": ("leaves", "left"),
    "come": ("comes", "came"), "speak": ("speaks", "spoke"),
    "work": ("works", "worked"), "want": ("wants", "wanted"),
    "need": ("needs", "needed"), "like": ("likes", "liked"),
    "use": ("uses", "used"), "help": ("helps", "helped"),
    "have": ("has", "had"), "be": ("is", "was"),
}
PARTICIPLES = {"gone": "went", "seen": "saw", "taken": "took",
               "given": "gave", "written": "wrote", "eaten": "ate",
               "known": "knew", "spoken": "spoke", "been": "was"}
# Fixed rare-first ordering, following DaLA; based on EWT development coverage.
RULES = ("perfect_participle", "do_support_form", "modal_verb_form",
         "demonstrative_number", "subject_verb_agreement", "subject_pronoun_case")


def candidates(row):
    """Return all licensed edits; abstain when source/alignment is suspect."""
    words = row["words"]
    if not row["aligned"] or any(
        w["feats"].get("Typo") == "Yes" or "CorrectForm" in w["misc"]
        or w["feats"].get("Foreign") == "Yes" for w in words
    ):
        return []
    by_id = {w["id"]: w for w in words}
    children = {w["id"]: [] for w in words}
    for w in words:
        if w["head"] in children:
            children[w["head"]].append(w)
    edits = []

    def add(rule, word, replacement):
        if word["span"] is None or not word["form"].isalpha():
            return
        original = word["form"]
        if original.isupper() and len(original) > 1:
            replacement = replacement.upper()
        elif original[0].isupper() and (original != "I" or not row["doc"][:word["span"][0]].strip(" \t\n\"'([{")):
            replacement = replacement.capitalize()
        if replacement != original:
            edits.append(Edit(rule, word["id"], *word["span"], original, replacement))

    for w in words:
        form = w["form"].lower()
        dependents = children[w["id"]]
        head = by_id.get(w["head"])
        # Only immediately adjacent demonstrative + ordinary noun. Excludes
        # coordination, proper names, measure phrases and attributive compounds.
        if (form in {"this", "that", "these", "those"} and w["upos"] == "DET"
                and w["deprel"] == "det" and head and head["upos"] == "NOUN"
                and int(head["id"]) == int(w["id"]) + 1
                and not any(c["deprel"].split(":")[0] in {"conj", "nummod", "compound"}
                            for c in children[head["id"]])):
            number = head["feats"].get("Number")
            if (form in {"this", "that"} and number == "Sing"
                    or form in {"these", "those"} and number == "Plur"):
                add("demonstrative_number", w,
                    {"this": "these", "that": "those", "these": "this", "those": "that"}[form])

        auxiliaries = [c for c in dependents if c["deprel"] == "aux"]
        if w["upos"] in {"VERB", "AUX"}:
            for aux in auxiliaries:
                # Permit attached intervening adverbs (e.g. did not go,
                # has already eaten), but no intervening verbs or clauses.
                if int(aux["id"]) >= int(w["id"]):
                    continue
                between = [t for t in words if int(aux["id"]) < int(t["id"]) < int(w["id"])]
                if any(t["deprel"] != "advmod" or t["head"] != w["id"]
                       or t["upos"] not in {"ADV", "PART"} for t in between):
                    continue
                if w["xpos"] == "VB" and form in BASE_FORMS:
                    if aux["form"].lower() in {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}:
                        add("modal_verb_form", w, BASE_FORMS[form][0])
                    elif aux["form"].lower() in {"do", "does", "did"}:
                        add("do_support_form", w, BASE_FORMS[form][1])
                if (w["xpos"] == "VBN" and form in PARTICIPLES
                        and aux["lemma"] == "have" and aux["form"].lower() in {"have", "has", "had"}):
                    add("perfect_participle", w, PARTICIPLES[form])

        # Pronoun subjects avoid collective-noun/notional agreement and singular
        # 'they'. Only simple, uncoordinated clauses with an overt finite verb.
        if (w["upos"] != "PRON" or w["deprel"] != "nsubj"
                or form not in {"i", "he", "she", "we", "they"} or not head
                or dependents or int(w["id"]) >= int(head["id"])):
            continue
        clause_children = children[head["id"]]
        if any(c["deprel"].split(":")[0] in {"conj", "mark"} for c in clause_children):
            continue
        finite = [v for v in [head] + clause_children
                  if v["upos"] in {"VERB", "AUX"}
                  and v["feats"].get("VerbForm") == "Fin"
                  and v["feats"].get("Mood") == "Ind"
                  and (v is head or v["deprel"] in {"aux", "cop"})]
        if len(finite) != 1:
            continue
        verb = finite[0]
        # Adjacency excludes ellipsis, parentheticals and comparative fragments.
        if int(verb["id"]) != int(w["id"]) + 1:
            continue
        add("subject_pronoun_case", w,
            {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}[form])
        v = verb["form"].lower()
        expected = ({"am": "is", "have": "has", "do": "does"} if form == "i"
                    else {"is": "are", "has": "have", "does": "do"} if form in {"he", "she"}
                    else {"are": "is", "were": "was", "have": "has", "do": "does"})
        if v in expected:
            add("subject_verb_agreement", verb, expected[v])
        elif verb["upos"] == "VERB" and verb["lemma"] in BASE_FORMS:
            base = verb["lemma"]
            third = BASE_FORMS[base][0]
            if form in {"he", "she"} and v == third and verb["xpos"] == "VBZ":
                add("subject_verb_agreement", verb, base)
            elif form in {"i", "we", "they"} and v == base and verb["xpos"] == "VBP":
                add("subject_verb_agreement", verb, third)
    return edits


def choose_edit(row, seed=4242):
    """First eligible type in rare-first order; seeded choice among its spans."""
    edits = candidates(row)
    if not edits:
        return None
    types = {e.corruption_type for e in edits}
    digest = hashlib.sha256(f"{seed}\0{row['doc']}".encode()).digest()
    selected = next(rule for rule in RULES if rule in types)
    matches = [e for e in edits if e.corruption_type == selected]
    return matches[int.from_bytes(digest[8:16], "big") % len(matches)]
