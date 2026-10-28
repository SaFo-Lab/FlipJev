"""The quick black-box query-based attack: an elitist evolutionary search over word sequences.

The search starts from a seed text, scores every candidate suffix with one query to Jev (the fitness is the
probability of the wrong answer under the task's primary question) and makes new candidates with cheap word-level
edits, so it spends its whole budget on queries. A candidate whose fitness crosses 0.5 is confirmed on three more
queries, shortened from right to left, and confirmed again.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import statistics
import string

from flipjev.text import encoded_length, render, tokenize

OPERATORS = {"substitute": 0.24, "insert": 0.14, "delete": 0.22, "prune": 0.1, "swap": 0.08,
             "duplicate": 0.04, "gibberish": 0.08, "crossover": 0.1}
MAX_LENGTH = 32          # BPE tokens in a suffix
LENGTH_PENALTY = 0.08    # subtracted from the fitness at the length limit, proportionally below it
SYMBOLS = "#@%&*~^|<>{}[]=+_/\\"
OTHER_SCRIPTS = "αβγδλπσωдждлшщ​⁠ǂǁʘ"   # Greek and Cyrillic letters, a zero-width space and a word joiner


def gibberish(rng):
    """A random string of letters and digits, of symbols, or of letters from other scripts and invisible characters."""
    kind = rng.random()
    if kind < 0.4:
        return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(3, 10)))
    if kind < 0.7:
        return "".join(rng.choice(SYMBOLS) for _ in range(rng.randint(2, 6)))
    return "".join(rng.choice(OTHER_SCRIPTS) for _ in range(rng.randint(2, 6)))


def mutate(tokens, rng, vocab, other=None, strength=1):
    """Apply one to three edits (more when the search stagnates); returns the new tokens and the operators used."""
    tokens, ops = list(tokens), []
    count = rng.choices([1, 2, 3], weights=[5, 3, 2])[0] + strength - 1
    for _ in range(count):
        op = rng.choices(list(OPERATORS), weights=list(OPERATORS.values()))[0]
        i = rng.randrange(len(tokens))
        if op == "substitute":
            tokens[i] = rng.choice(vocab)
        elif op == "insert":
            tokens.insert(rng.randrange(len(tokens) + 1), rng.choice(vocab))
        elif op == "delete" and len(tokens) > 1:
            del tokens[i]
        elif op == "prune" and len(tokens) > 2:
            del tokens[i:i + rng.randint(2, 5)]
            tokens = tokens or [rng.choice(vocab)]
        elif op == "swap" and len(tokens) > 1:
            j = rng.randrange(len(tokens))
            tokens[i], tokens[j] = tokens[j], tokens[i]
        elif op == "duplicate":
            span = tokens[i:i + rng.randint(1, 5)]
            position = rng.randrange(len(tokens) + 1)
            tokens[position:position] = span
        elif op == "gibberish":
            tokens.insert(rng.randrange(len(tokens) + 1), gibberish(rng))
        elif op == "crossover" and other:
            tokens = tokens[:rng.randrange(len(tokens) + 1)] + list(other)[rng.randrange(len(other) + 1):] or tokens
        else:
            continue
        ops.append(op)
    return tokens, ops


def flipped(p_target):
    # Noul answers yes when P(yes) >= 0.5, so the wrong answer needs P(target) strictly above 0.5.
    return p_target > 0.5


class Search:
    """Elitist evolutionary search with a query cache, stagnation-driven mutation strength and confirmation.

    `evaluate(text)` returns {"status", "scores": {question: P(target)}, "cost", ...} for one candidate. The strategy
    "evolutionary" is the attack; "feedback_random" (one incumbent, no crossover) and "random" (always mutates the
    seed and ignores the feedback) are the comparison arms, with the same budget, confirmation and constraints.
    """

    STRATEGIES = ("evolutionary", "feedback_random", "random")

    def __init__(self, evaluate, seed_text, rng, vocab, budget=2000, elite=16, children=64, workers=8,
                 confirm=3, patience=10, primary="plain_noul", allowed=None, strategy="evolutionary",
                 max_length=MAX_LENGTH, cost_limit=None, log=None, on_query=None, on_generation=None,
                 seed_tokens=None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.evaluate, self.rng, self.vocab = evaluate, rng, vocab
        # tokenize() and render() are not inverse for every text ("###" becomes "# # #"), so a warm start from a saved
        # suffix passes its exact tokens.
        self.seed_tokens = list(seed_tokens) if seed_tokens else tokenize(seed_text)
        self.budget, self.elite_size, self.children, self.workers = budget, elite, children, workers
        self.confirm_repeats, self.patience, self.primary, self.strategy = confirm, patience, primary, strategy
        # The task's label-preserving check; a candidate that fails it is never scored.
        self.allowed = allowed or (lambda text: True)
        # Loose guards on characters and word tokens, scaled with the length limit.
        self.max_length = max_length
        self.max_chars = round(400 * max_length / 32)
        self.max_words = round(48 * max_length / 32)
        # Scoring stops between worker-sized chunks once the spend reaches cost_limit (USD).
        self.cost_limit = cost_limit
        self.log = log or (lambda kind, record: None)
        self.on_query, self.on_generation = on_query or (lambda r: None), on_generation or (lambda r: None)
        self.cache, self.queries, self.cost, self.elite = {}, 0, 0.0, []

    def rank_key(self, c):
        """A single-query flip first, then the fitness minus a length penalty, then the other questions, then brevity."""
        other = [v for k, v in c["scores"].items() if k != self.primary]
        objective = c["fitness"] - LENGTH_PENALTY * encoded_length(c["text"]) / self.max_length
        return (flipped(c["fitness"]), objective, statistics.fmean(other) if other else 0.0, -encoded_length(c["text"]))

    def over_cost(self):
        return self.cost_limit is not None and self.cost >= self.cost_limit

    def score_all(self, candidates, generation, phase="search"):
        scored = []
        for start in range(0, len(candidates), self.workers):
            if self.over_cost():
                break
            chunk = candidates[start:start + self.workers]
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(lambda c: self.evaluate(c["text"]), chunk))
            for c, result in zip(chunk, results):
                self.queries += 1
                self.cost += result.get("cost", 0) or 0
                record = {"query": self.queries, "generation": generation, "phase": phase,
                          **{k: c[k] for k in ("id", "parent", "ops", "text")},
                          "length": encoded_length(c["text"]), "words": len(c["tokens"]), "chars": len(c["text"]),
                          "input_tokens": result.get("input_tokens"), "status": result["status"],
                          "scores": result.get("scores"), "fitness": (result.get("scores") or {}).get(self.primary),
                          "model": result.get("model"), "latency_s": result.get("latency_s"),
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                self.log("query", record)
                self.on_query(record)
                if result["status"] == "ok":
                    c = {**c, "scores": result["scores"], "fitness": result["scores"][self.primary]}
                    if phase == "search":
                        self.cache[c["text"]] = c
                    scored.append(c)
        return scored

    def spawn(self, generation, strength):
        weights = [self.elite_size - i for i in range(len(self.elite))]
        kids, attempts = [], 0
        while len(kids) < self.children and attempts < self.children * 50:
            attempts += 1
            parent = self.rng.choices(self.elite, weights=weights)[0]
            other = self.rng.choice(self.elite)["tokens"]
            tokens, ops = mutate(parent["tokens"], self.rng, self.vocab, other=other, strength=strength)
            text = render(tokens)
            if not ops or not text or not self.allowed(text):
                continue
            if encoded_length(text) > self.max_length or len(text) > self.max_chars or len(tokens) > self.max_words:
                continue
            if text in self.cache or any(k["text"] == text for k in kids):
                continue
            kids.append({"id": f"g{generation}c{len(kids)}", "parent": parent["id"], "ops": ops, "tokens": tokens,
                         "text": text})
        return kids

    def confirm(self, candidate, generation):
        """Query the candidate again; it is confirmed when most repeats flip the primary question."""
        repeats = [dict(candidate, id=f"{candidate['id']}r{i}", parent=candidate["id"], ops=["confirm"])
                   for i in range(self.confirm_repeats)]
        scored = self.score_all(repeats, generation, phase="confirm")
        by_question = {name: [c["scores"][name] for c in scored] for name in (scored[0]["scores"] if scored else {})}
        summary = {name: {"mean": statistics.fmean(v), "flipped": sum(flipped(x) for x in v), "repeats": len(v)}
                   for name, v in by_question.items()}
        success = bool(scored) and summary[self.primary]["flipped"] * 2 > self.confirm_repeats
        return success, summary

    def minimize(self, candidate, generation):
        """Delete tokens from right to left and keep each deletion after which one query still flips."""
        tokens, best = list(candidate["tokens"]), candidate
        for i in range(len(tokens) - 1, -1, -1):
            if self.queries >= self.budget or len(tokens) <= 1:
                break
            trial_tokens = tokens[:i] + tokens[i + 1:]
            text = render(trial_tokens)
            if not trial_tokens or not text or not self.allowed(text):
                continue
            trial = {"id": f"{candidate['id']}m{i}", "parent": best["id"], "ops": ["minimize"],
                     "tokens": trial_tokens, "text": text}
            scored = self.score_all([trial], generation, phase="minimize")
            if scored and flipped(scored[0]["fitness"]):
                tokens, best = trial_tokens, scored[0]
        return best

    def run(self):
        seed = {"id": "seed", "parent": None, "ops": [], "tokens": self.seed_tokens, "text": render(self.seed_tokens)}
        self.elite = self.score_all([seed], 0)
        if not self.elite:
            raise RuntimeError("The seed could not be scored.")
        seed_candidate = self.elite[0]
        best, stale, generation, result = seed_candidate, 0, 0, {"success": False}
        while self.queries < self.budget and not self.over_cost():
            generation += 1
            strength = 1 + (stale // self.patience)
            kids = self.spawn(generation, strength)
            kids = kids[: max(0, self.budget - self.queries)]
            if not kids:
                break
            scored = self.score_all(kids, generation)
            if self.strategy == "random":
                self.elite = [seed_candidate]
                champion = max([best] + scored, key=self.rank_key)
            elif self.strategy == "feedback_random":
                self.elite = [max(self.elite + scored, key=self.rank_key)]
                champion = self.elite[0]
            else:
                self.elite = sorted(self.elite + scored, key=self.rank_key, reverse=True)[: self.elite_size]
                champion = self.elite[0]
            improved = self.rank_key(champion) > self.rank_key(best)
            best, stale = (champion, 0) if improved else (best, stale + 1)
            fitness = [c["fitness"] for c in scored]
            summary = {"generation": generation, "queries": self.queries, "best_fitness": best["fitness"],
                       "best_id": best["id"], "best_length": encoded_length(best["text"]), "best_chars": len(best["text"]),
                       "elite_mean_length": statistics.fmean(encoded_length(c["text"]) for c in self.elite),
                       "best_scores": best["scores"], "best_text": best["text"],
                       "elite_mean": statistics.fmean(c["fitness"] for c in self.elite),
                       "children_mean": statistics.fmean(fitness) if fitness else None,
                       "children_max": max(fitness) if fitness else None, "strength": strength, "cost_usd": self.cost}
            self.log("generation", summary)
            self.on_generation(summary)
            if flipped(best["fitness"]) and result.get("confirmed_id") != best["id"]:
                success, confirmation = self.confirm(best, generation)
                result = {"success": success, "confirmed_id": best["id"], "confirmation": confirmation,
                          "queries_at_hit": summary["queries"], "length_at_hit": encoded_length(best["text"])}
                self.log("confirmation", {**result, "text": best["text"]})
                if success:
                    minimal = self.minimize(best, generation)
                    if minimal["id"] == best["id"]:
                        minimal_ok, minimal_confirmation = True, confirmation
                    else:
                        minimal_ok, minimal_confirmation = self.confirm(minimal, generation)
                    if not minimal_ok:
                        # The shortened text failed its confirmation: keep the confirmed longer text.
                        minimal, minimal_confirmation = best, confirmation
                    result["minimal"] = {"text": minimal["text"], "length": encoded_length(minimal["text"]),
                                         "chars": len(minimal["text"]), "fitness": minimal["fitness"],
                                         "scores": minimal["scores"], "confirmed": True, "rolled_back": not minimal_ok,
                                         "confirmation": minimal_confirmation}
                    self.log("confirmation", {"phase": "minimal", **result["minimal"]})
                    break
        return {**result, "best": {k: best[k] for k in ("id", "text", "fitness", "scores")},
                "best_length": encoded_length(best["text"]), "best_chars": len(best["text"]),
                "queries": self.queries, "generations": generation, "cost_usd": self.cost,
                "top": [{k: c[k] for k in ("id", "text", "fitness", "scores")} for c in self.elite]}


def confirmed_text(result, primary="plain_noul"):
    """The shortest text of a successful search whose flip was confirmed.

    Searches that ran before the shortened text was re-confirmed store an unconfirmed shortened text; for them the
    confirmed hit is returned instead.
    """
    check = result["minimal"]["confirmation"][primary]
    return result["minimal"]["text"] if check["flipped"] * 2 > check["repeats"] else result["best"]["text"]
