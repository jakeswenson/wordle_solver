use linfa::prelude::Fit;
use linfa::Dataset;
use linfa_trees::SplitQuality;
use std::collections::{HashMap, HashSet};

#[derive(Default, Clone, Debug)]
struct GuessState {
    exact: HashMap<usize, char>,
    exactly_not: HashMap<usize, HashSet<char>>,
    must_contain: HashSet<char>,
    does_not_contain: HashSet<char>,
}

impl GuessState {
    fn parse(state: &str) -> GuessState {
        let guesses: Vec<_> = state.split(',').collect();

        guesses.iter().fold(GuessState::default(), |state, word| {
            let mut char_iter = word.chars().into_iter();
            let mut idx = 0;

            let mut state = state;

            while let Some(c) = char_iter.next() {
                match c {
                    '?' => {
                        let next = char_iter.next().unwrap().to_lowercase().nth(0).unwrap();
                        state.must_contain.insert(next);
                        state.exactly_not.entry(idx).or_default().insert(next);
                    }
                    c if c.is_uppercase() => {
                        state.exact.insert(idx, c.to_lowercase().nth(0).unwrap());
                    }
                    c => {
                        state.does_not_contain.insert(c);
                        state
                            .exactly_not
                            .entry(idx)
                            .or_default()
                            .insert(c.to_lowercase().nth(0).unwrap());
                    }
                }

                idx += 1;
            }

            return state;
        })
    }

    fn matches(self, word: &str) -> bool {
        return word.char_indices().all(|(idx, c)| {
            if let Some(a) = self.exact.get(&idx) {
                return if *a == c { true } else { false };
            } else if let Some(not) = self.exactly_not.get(&idx) {
                if not.contains(&c) {
                    return false;
                }
            }

            if (&self.does_not_contain).contains(&c) {
                return false;
            }

            return true;
        }) && self.must_contain.iter().all(|c| word.contains(*c));
    }
}

fn sort_letters(word: &str) -> String {
    let mut chars = word.chars().collect::<Vec<_>>();
    chars.sort();

    let word: String = chars.iter().cloned().collect();

    return word;
}

fn main() {
    let words = include_str!("../words.txt").lines();
    let mut map: HashMap<String, HashSet<String>> = HashMap::new();

    let mut args = std::env::args();
    args.next();
    let letters = args.next().unwrap_or("".to_string());
    let guess = GuessState::parse(&letters);

    println!("{:?}", guess);

    println!("letters: {}", letters);

    words.clone().for_each(|word| {
        let chars = sort_letters(word);
        if guess.clone().matches(word) {
            map.entry(chars)
                .or_insert(HashSet::new())
                .insert(word.to_string());
        }
    });

    let mut keys: Vec<_> = map.keys().cloned().collect();

    keys.sort_by_key(|k| map.get(k.as_str()).map(|v| v.len()).unwrap_or(0));

    println!(
        "Words: {} {:?}",
        keys.len(),
        keys[keys.len() - std::cmp::min(10, keys.len())..]
            .iter()
            .map(|k| {
                let word = map.get(k).unwrap();
                if word.len() == 1 {
                    return (word.iter().cloned().next().unwrap(), 1);
                }
                return (k.clone(), word.len());
            })
            .collect::<Vec<_>>()
    );

    let first = keys.last().unwrap();

    let all_words: Vec<_> = words.filter(|w| guess.clone().matches(w)).collect();

    predict(&map, &all_words);

    println!("Word {} {}", first, map.get(first).unwrap().len());

    let words = map.get(first).unwrap();

    println!(
        "{:?}",
        words
            .into_iter()
            .filter(|f| guess.clone().matches(f))
            .collect::<Vec<_>>()
    );
}

fn predict(map: &HashMap<String, HashSet<String>>, all_words: &Vec<&str>) {
    let letters = all_words
        .iter()
        .fold(HashSet::new(), |s, a| {
            a.chars().fold(s, |s, a| {
                let mut set = s;
                set.insert(a);
                return set;
            })
        })
        .into_iter()
        .collect::<Vec<_>>();

    println!("All letters: {:?}", letters);

    let word_features = all_words
        .iter()
        .flat_map(|w| {
            letters
                .iter()
                .cloned()
                .map(|l| if w.contains(l) { 1f64 } else { 0f64 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let dataset: Dataset<f64, usize> = Dataset::new(
        ndarray::Array::from_shape_vec((all_words.len(), letters.len()), word_features).unwrap(),
        ndarray::arr1(
            &all_words
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .collect::<Vec<usize>>(),
        ),
    )
    .with_feature_names(letters.iter().map(|l| format!("has_{}", l)).collect());

    let result = linfa_trees::DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .fit(&dataset)
        .unwrap();

    dbg!(result
        .features()
        .into_iter()
        .map(|i| dataset.feature_names()[i].clone())
        .collect::<Vec<_>>());
    dbg!(result.feature_importance());

    let (feature_idx, _, _) = dbg!(result.root_node().split());
    dbg!(result.root_node().feature_name());

    let next_best = all_words
        .iter()
        .cloned()
        .filter(|w| w.contains(letters[feature_idx]))
        .filter(|w| {
            let letters = sort_letters(w);
            let unique_letters = letters.chars().collect::<HashSet<_>>().len();
            letters.len() == unique_letters
        })
        .max_by_key(|f| {
            let letters = sort_letters(f);
            let num_words = map.get(&letters).map(|s| s.len()).unwrap_or(0);
            return num_words;
        });

    println!("Prediction: {:?}", next_best);
}
