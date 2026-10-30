from collections import Counter, defaultdict

import regex

PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)|"""
    r""" ?\p{L}+|"""
    r""" ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)

def _merge_word(word, pair):
    left, right = pair
    merged = left + right
    result = []
    index = 0

    while index < len(word):
        if (
            index + 1 < len(word)
            and word[index] == left
            and word[index + 1] == right
        ):
            result.append(merged)
            index += 2
        else:
            result.append(word[index])
            index += 1
    return tuple(result)

def _count_pretokens(text, special_tokens):
    word_counts = Counter()
    if special_tokens:
        escaped_tokens = sorted(
            (regex.escape(token) for token in special_tokens),
            key=len,
            reverse=True,
        )
        special_pattern = "|".join(escaped_tokens)
        chunks = regex.split(special_pattern, text)

    else:
        chunks = [text]
    for chunk in chunks:
        if not chunk:
            continue
        for match in regex.finditer(PRETOKEN_PATTERN, chunk):
            piece = match.group()
            encoded = piece.encode("utf-8")

            tokens = tuple(
                encoded[index : index + 1]
                for index in range(len(encoded))
            )
            word_counts[tokens] += 1

    return word_counts

def train_bpe(input_path, vocab_size, special_tokens=None):
    special_tokens = special_tokens or []

    with open(input_path, encoding="utf-8") as file:
        text = file.read()

    word_counts = _count_pretokens(text, special_tokens)

    vocab = {
        index: bytes([index])
        for index in range(256)
    }

    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")

        if special_bytes not in vocab.values():
            vocab[len(vocab)] = special_bytes

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than "
            f"the initial vocabulary size {len(vocab)}"
        )

    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word, frequency in word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency
            pair_to_words[pair].add(word)

    merges = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )

        if best_count <= 0:
            break

        affected_words = list(pair_to_words.get(best_pair, ()))

        for word in affected_words:
            frequency = word_counts.pop(word, 0)
            if frequency == 0:
                continue

            for old_pair in zip(word, word[1:]):
                pair_counts[old_pair] -= frequency
                pair_to_words[old_pair].discard(word)

                if pair_counts[old_pair] <= 0:
                    pair_counts.pop(old_pair, None)
                    pair_to_words.pop(old_pair, None)

            new_word = _merge_word(word, best_pair)
            word_counts[new_word] += frequency

            for new_pair in zip(new_word, new_word[1:]):
                pair_counts[new_pair] += frequency
                pair_to_words[new_pair].add(new_word)

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        if merged_token not in vocab.values():
            vocab[len(vocab)] = merged_token
        else:
            vocab[len(vocab)] = merged_token

    return vocab, merges