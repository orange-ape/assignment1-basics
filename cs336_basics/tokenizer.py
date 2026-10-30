import json

import regex


PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)|"""
    r""" ?\p{L}+|"""
    r""" ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)

PRETOKEN_REGEX = regex.compile(PRETOKEN_PATTERN)


def _bytes_to_unicode():
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]

    next_code_point = 256
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(next_code_point)
            next_code_point += 1

    return {
        byte_value: chr(code_point)
        for byte_value, code_point in zip(byte_values, code_points)
    }


class Tokenizer:
    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        next_id = max(self.vocab, default=-1) + 1

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes not in self.token_to_id:
                self.vocab[next_id] = special_bytes
                self.token_to_id[special_bytes] = next_id
                next_id += 1

        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

        self.merge_ranks = {}

        for rank, pair in enumerate(self.merges):
            if pair not in self.merge_ranks:
                self.merge_ranks[pair] = rank

        if self.special_tokens:
            ordered_specials = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )
            special_pattern = "|".join(
                regex.escape(token)
                for token in ordered_specials
            )
            self.special_regex = regex.compile(special_pattern)
        else:
            self.special_regex = None

        self._cache = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None,
    ):
        with open(vocab_filepath, encoding="utf-8") as file:
            encoded_vocab = json.load(file)

        bytes_to_unicode = _bytes_to_unicode()
        unicode_to_bytes = {
            character: byte_value
            for byte_value, character in bytes_to_unicode.items()
        }

        vocab = {
            token_id: bytes(
                unicode_to_bytes[character]
                for character in token
            )
            for token, token_id in encoded_vocab.items()
        }

        merges = []

        with open(merges_filepath, encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip().split()

                if len(parts) != 2:
                    continue

                left, right = parts
                merges.append(
                    (
                        bytes(
                            unicode_to_bytes[character]
                            for character in left
                        ),
                        bytes(
                            unicode_to_bytes[character]
                            for character in right
                        ),
                    )
                )

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _merge_pair(self, tokens, pair):
        left, right = pair
        result = []
        index = 0

        while index < len(tokens):
            if (
                index + 1 < len(tokens)
                and tokens[index] == left
                and tokens[index + 1] == right
            ):
                result.append(left + right)
                index += 2
            else:
                result.append(tokens[index])
                index += 1

        return result

    def _encode_pretoken(self, pretoken, cache):
        if cache is not None and pretoken in cache:
            return cache[pretoken]

        if pretoken in self.token_to_id:
            result = (self.token_to_id[pretoken],)

            if cache is not None:
                cache[pretoken] = result

            return result

        tokens = [
            pretoken[index : index + 1]
            for index in range(len(pretoken))
        ]

        while len(tokens) > 1:
            best_pair = None
            best_rank = None

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = self._merge_pair(tokens, best_pair)

        result = tuple(
            self.token_to_id[token]
            for token in tokens
        )

        if cache is not None:
            cache[pretoken] = result

        return result

    def _encode_ordinary(self, text, cache):
        token_ids = []

        for match in PRETOKEN_REGEX.finditer(text):
            pretoken = match.group().encode("utf-8")
            token_ids.extend(
                self._encode_pretoken(pretoken, cache)
            )

        return token_ids

    def _encode_text(self, text, cache):
        token_ids = []

        if self.special_regex is None:
            return self._encode_ordinary(text, cache)

        cursor = 0

        for match in self.special_regex.finditer(text):
            ordinary_text = text[cursor : match.start()]
            token_ids.extend(
                self._encode_ordinary(ordinary_text, cache)
            )

            special_token = match.group()
            token_ids.append(self.special_to_id[special_token])
            cursor = match.end()

        token_ids.extend(
            self._encode_ordinary(text[cursor:], cache)
        )

        return token_ids

    def encode(self, text):
        return self._encode_text(text, self._cache)

    def encode_iterable(self, iterable):
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line += "\n"

                yield from self._encode_text(line, cache=None)

        if buffer:
            yield from self._encode_text(buffer, cache=None)

    def decode(self, ids):
        data = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )
        return data.decode("utf-8", errors="replace")