//! Search fusion shared by STM and LTM recall: the keyword-query builder and
//! reciprocal rank fusion, both configured by [`Ranking`] (`[retrieval]`).
//!
//! Each leg (vector, keyword) ranks its own candidates; a candidate's fused
//! score is `Σ weight / (rrf_k + rank)` over the legs it appears in, with
//! 1-based ranks. A rank, unlike a raw vector distance or BM25 value, means
//! the same in both legs.

use super::tuning::Ranking;
use std::collections::HashMap;
use std::hash::Hash;

/// English stop-words dropped from the keyword query (sorted, lowercase).
/// They match nearly every stored text, so as OR-ed FTS terms they let any
/// chatty entry outrank the right one.
pub const STOP_WORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// Most terms one keyword query carries (bounds pathological inputs).
const MAX_QUERY_TERMS: usize = 32;

fn is_stop_word(term: &str) -> bool {
    STOP_WORDS
        .binary_search(&term.to_lowercase().as_str())
        .is_ok()
}

/// Turn raw query text into a **safe** FTS5 MATCH expression, or `None` if it
/// carries no searchable terms (then the keyword leg is skipped).
///
/// The raw text is split on non-alphanumerics and each token quoted as a
/// phrase, which neutralises FTS operators and punctuation (a raw `:`, `-` or
/// `"` would be a syntax error), and the tokens are OR-ed, so one absent word
/// doesn't empty the result. With `drop_stop_words`, English stop-words are
/// left out unless the query is made of nothing else.
pub fn fts_match_query(raw: &str, drop_stop_words: bool) -> Option<String> {
    let tokens: Vec<&str> = raw
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let content: Vec<&str> = if drop_stop_words {
        tokens
            .iter()
            .copied()
            .filter(|t| !is_stop_word(t))
            .collect()
    } else {
        Vec::new()
    };
    let terms: Vec<String> = if content.is_empty() { tokens } else { content }
        .into_iter()
        .take(MAX_QUERY_TERMS)
        .map(|t| format!("\"{t}\""))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

/// How many candidates each leg should rank for a final top `k`.
pub fn leg_depth(k: usize, ranking: &Ranking) -> usize {
    k.max(1).saturating_mul(ranking.leg_k_factor.max(1))
}

/// Fuse two best-first candidate lists by reciprocal rank fusion. Returns
/// every candidate once, best first, with its fused score. Ties keep the
/// better vector rank, then the better keyword rank, so the order is
/// deterministic. A leg with weight 0 still contributes candidates, but no
/// score. Duplicates within one list count at their first (best) rank.
pub fn rrf_fuse<K: Eq + Hash + Clone>(
    vector: &[K],
    keyword: &[K],
    ranking: &Ranking,
) -> Vec<(K, f64)> {
    #[derive(Default)]
    struct Entry {
        score: f64,
        vector_rank: Option<usize>,
        keyword_rank: Option<usize>,
    }
    let mut entries: HashMap<K, Entry> = HashMap::new();
    let mut order: Vec<K> = Vec::new();
    let mut add = |list: &[K], weight: f64, is_vector: bool| {
        for (i, key) in list.iter().enumerate() {
            let rank = i + 1;
            let entry = entries.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                Entry::default()
            });
            let slot = if is_vector {
                &mut entry.vector_rank
            } else {
                &mut entry.keyword_rank
            };
            if slot.is_none() {
                *slot = Some(rank);
                entry.score += weight / (ranking.rrf_k + rank as f64);
            }
        }
    };
    add(vector, ranking.vector_weight, true);
    add(keyword, ranking.keyword_weight, false);

    let mut fused: Vec<(K, Entry)> = order
        .into_iter()
        .map(|k| {
            let e = entries.remove(&k).expect("every ordered key has an entry");
            (k, e)
        })
        .collect();
    fused.sort_by(|(_, a), (_, b)| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| rank_key(a.vector_rank).cmp(&rank_key(b.vector_rank)))
            .then_with(|| rank_key(a.keyword_rank).cmp(&rank_key(b.keyword_rank)))
    });
    fused.into_iter().map(|(k, e)| (k, e.score)).collect()
}

/// An absent rank sorts after every present one.
fn rank_key(rank: Option<usize>) -> usize {
    rank.unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_words_are_sorted() {
        assert!(STOP_WORDS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn match_query_is_safe_and_drops_stop_words() {
        assert_eq!(
            fts_match_query("What did I say about the lighthouse?", true).as_deref(),
            Some("\"say\" OR \"lighthouse\"")
        );
        assert_eq!(
            fts_match_query("What did I say", false).as_deref(),
            Some("\"What\" OR \"did\" OR \"I\" OR \"say\"")
        );
        // Only stop-words: kept, so the query still searches.
        assert_eq!(
            fts_match_query("what is it", true).as_deref(),
            Some("\"what\" OR \"is\" OR \"it\"")
        );
        // Operators and quotes are neutralised.
        assert_eq!(
            fts_match_query("a\"b NEAR(c) -d", true).as_deref(),
            Some("\"b\" OR \"NEAR\" OR \"c\" OR \"d\"")
        );
        assert_eq!(fts_match_query("  ?! ", true), None);
        let long = "word ".repeat(100);
        assert_eq!(
            fts_match_query(&long, true).unwrap().matches("OR").count(),
            MAX_QUERY_TERMS - 1
        );
    }

    #[test]
    fn rrf_rewards_agreement_and_respects_weights() {
        let r = Ranking::default(); // rrf_k 60, weights 1:1
        let fused = rrf_fuse(&["a", "b", "c"], &["c", "d"], &r);
        let keys: Vec<&str> = fused.iter().map(|(k, _)| *k).collect();
        // c is in both legs: 1/63 + 1/61 beats a's 1/61.
        assert_eq!(keys, ["c", "a", "b", "d"]);
        assert!((fused[0].1 - (1.0 / 63.0 + 1.0 / 61.0)).abs() < 1e-12);
        // b (vector rank 2) and d (keyword rank 2) tie at 1/62: vector first.
        assert!((fused[2].1 - fused[3].1).abs() < 1e-12);

        let keyword_only = Ranking {
            vector_weight: 0.0,
            ..Ranking::default()
        };
        let fused = rrf_fuse(&["a", "b"], &["b", "c"], &keyword_only);
        let keys: Vec<&str> = fused.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            keys,
            ["b", "c", "a"],
            "a has no score but stays a candidate"
        );
        assert_eq!(fused[2].1, 0.0);
    }

    #[test]
    fn rrf_counts_a_duplicate_once_and_handles_empty_legs() {
        let r = Ranking::default();
        let fused = rrf_fuse(&["a", "a", "b"], &[], &r);
        assert_eq!(fused.len(), 2);
        assert!((fused[0].1 - 1.0 / 61.0).abs() < 1e-12);
        assert!(rrf_fuse::<&str>(&[], &[], &r).is_empty());
        assert_eq!(leg_depth(10, &r), 40);
        assert_eq!(leg_depth(0, &r), 4);
    }
}
