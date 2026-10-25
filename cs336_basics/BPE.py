"""
To dos: 1) reimplement encode yourself
2) Major inefficienty in that we run max() on every pair merge
"""


import regex as re
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time
import pickle
from pathlib import Path
import os
import math
from functools import partial


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

test = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"


# def pre_tokenize(text):
#     """
#     Given a text, it pretokenizes the text. It then counts up word frequencies and 
#     returns them as a Counter object of 
#     """
#     counts = Counter()
#     for pretoken in re.finditer(PAT, text):
#         bytes_pre_token = pretoken.group().encode("utf-8")
#         counts[bytes_pre_token] += 1
#     return counts


def split_on_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in special_tokens)
    pieces = re.split(f"({pattern})", text)
    return [piece for piece in pieces if piece]


def pre_tokenize(text, special_tokens):
    """
    Takes text and a list of special tokens
    Returns a dictionary with tuples (representing different words/chunks from the text) 
    and their count in the text
    final output is of form
    {.....
    (b"c", b"a", b"t") : 10_000
    .....
    }
    """

    #strings is a list of stories seperated by special tokens
    strings = split_on_special_tokens(text, special_tokens)
    final_count = Counter()
    for str in strings:
        if str not in special_tokens:    
            chunks = re.findall(PAT, str) #Returns a list of chunks we can vocab on
            byte_chunks = [tuple(bytes([b]) for b in chunk.encode("utf-8")) for chunk in chunks] #List of chunks where each chunk is broken down into a tuple of single bytes

            counts = {}

            for sub in byte_chunks: #Sub is a tuple of bytes
                if sub in counts:
                    counts[sub] += 1
                else:
                    counts[sub] = 1
            
            final_count.update(counts)

    return final_count





def get_word_frequencies(file_path, special_tokens, num_processes=8):
    """
    Creates num_processes worker processes using multiprocessing where each process tokenizes a section of the file.
    Reads chunks through a generator and combines the pre-token frequencies returned by each process.
    Chunk boundaries occur at the first supplied special token. If no special tokens are supplied,
    the whole file is processed as one chunk. Special tokens are excluded from the counts.

    file_path: path to the UTF-8 text file to pre-tokenize
    special_tokens: list of special tokens that separate text and are excluded from counting
    num_processes: number of worker processes, defaults to 8

    Returns a Counter mapping tuples of single-byte bytes objects to their frequencies in the file.
    """
    output = Counter()
    target_chunk_bytes = 64 * 1024**2  # Approximately 64 MiB per chunk
    file_size = os.path.getsize(file_path)
    num_chunks = max(num_processes, math.ceil(file_size / target_chunk_bytes))
    worker = partial(pre_tokenize, special_tokens=special_tokens)
    
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, special_tokens[0].encode("utf-8")
        ) if special_tokens else [0, file_size]

        tasks = [
            (file_path, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        if not tasks:
            return output

        def chunk_texts():
            for _, start, end in tasks:
                f.seek(start)
                yield f.read(end - start).decode("utf-8", errors="ignore")

        with Pool(processes=num_processes) as pool:
            for counts in pool.imap(worker, chunk_texts()):
                output.update(counts)

    return output







def BPE_train(input_path, vocab_size, special_tokens):
    """
    Takes a file path to a file holding text and returns the vocabulary and 
    mergelist that comes out of this.
    """
    #Initialize the vocabulary and count the pre-tokenized words
    vocabulary = [token.encode("utf-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    merges = []
    word_frequencies = get_word_frequencies(input_path, special_tokens)

    #Build a dictionary of pairs and their counts and sources (what words they come from)
    # Of the form [count, {sources}]
    pairs = {}
    for word, count in word_frequencies.items():
        for pair in zip(word[:-1], word[1:]):
            if pair in pairs:
                val = pairs[pair]
                val[0] += word_frequencies[word]
                val[1].add(word)
            else:
                pairs[pair] = [word_frequencies[word], {word}]

    
    for _ in range(vocab_size-len(vocabulary)):
        if not pairs:
            break

        #Pick the most frequent pair, breaking ties lexicographically
        best_pair = max(pairs, key=lambda pair: (pairs[pair][0], pair))
        data = pairs[best_pair]
        new_token = best_pair[0] + best_pair[1]
        vocabulary.append(new_token)
        merges.append(best_pair)
        #Update only the words containing the selected pair
        sources = data[1]
        for source in sources.copy():
            count = word_frequencies[source]

            #Remove the old word's pair counts and source entries
            for pair in zip(source[:-1], source[1:]):
                pairs[pair][0] -= count
                pairs[pair][1].discard(source)
                if pairs[pair][0] == 0:
                    del pairs[pair]

            #Merge non-overlapping occurrences from left to right
            word = ()
            pointer = 0
            while pointer < len(source):
                char = source[pointer]
                if char != best_pair[0]:
                    word += (char,)
                    pointer += 1
                    continue
                else:
                    if pointer+1 < len(source) and source[pointer+1]== best_pair[1]:
                        word += (new_token,)
                        pointer += 2
                    else:
                        word += (char,)
                        pointer += 1

            #Replace the word key and add its updated pair counts and sources
            del word_frequencies[source]
            word_frequencies[word] += count
            for pair in zip(word[:-1], word[1:]):
                if pair in pairs:
                    pairs[pair][0] += count
                    pairs[pair][1].add(word)
                else:
                    pairs[pair] = [count, {word}]

    #Return token IDs mapped to bytes, and the ordered merges
    return {i:vocabulary[i] for i in range(len(vocabulary))}, merges
            



def BPE_train_and_save(input_path, vocab_size, special_tokens, vocab_filepath, merges_filepath):
    """
    Trains BPE and saves the vocabulary and ordered merges to separate pickle files.
    Returns the vocabulary and merges.
    """
    #Train the vocabulary and merges
    vocab, merges = BPE_train(input_path, vocab_size, special_tokens)

    #Create the output directories and save both files
    vocab_filepath = Path(vocab_filepath)
    merges_filepath = Path(merges_filepath)
    vocab_filepath.parent.mkdir(parents=True, exist_ok=True)
    merges_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_filepath, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_filepath, "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges


class Tokenizer():
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Loads the vocabulary and merges saved by BPE_train_and_save."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str)->list[int]:
        #Build the lookup dictionaries once
        if not hasattr(self, "merge_ranks"):
            self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
            self.lookup_vocab_index = {token: id for id, token in self.vocab.items()}
            for token in self.special_tokens:
                token = token.encode("utf-8")
                if token not in self.lookup_vocab_index:
                    id = len(self.vocab)
                    self.vocab[id] = token
                    self.lookup_vocab_index[token] = id

        IDs_list = []
        strings = split_on_special_tokens(text, self.special_tokens)
        for str in strings:
            #Keep special tokens as single IDs
            if str in self.special_tokens:
                IDs_list.append(self.lookup_vocab_index[str.encode("utf-8")])
                continue

            #Pre-tokenize ordinary text and split each chunk into bytes
            chunks = re.findall(PAT, str)
            for chunk in chunks:
                chunk = [bytes([b]) for b in chunk.encode("utf-8")]

                #Apply the earliest available merge until none remain
                while len(chunk) > 1:
                    pairs = [(chunk[i], chunk[i+1]) for i in range(len(chunk)-1)]
                    valid_pairs = [pair for pair in pairs if pair in self.merge_ranks]
                    if not valid_pairs:
                        break
                    best_pair = min(valid_pairs, key=self.merge_ranks.get)

                    new_chunk = []
                    pointer = 0
                    while pointer < len(chunk):
                        if pointer+1 < len(chunk) and (chunk[pointer], chunk[pointer+1]) == best_pair:
                            new_chunk.append(best_pair[0] + best_pair[1])
                            pointer += 2
                        else:
                            new_chunk.append(chunk[pointer])
                            pointer += 1
                    chunk = new_chunk

                #Convert the merged chunk to token IDs
                for token in chunk:
                    IDs_list.append(self.lookup_vocab_index[token])
        return IDs_list

    def encode_iterable(self, iterable):
        """
        Given an iterable (prolly a file) yeilds token ids from the corpus.
        """
        for line in iterable:
            ids = self.encode(line)
            for token in ids:
                yield token

    def decode(self, ids):
        byte_string = b"".join(self.vocab[i] for i in ids)
        return byte_string.decode("utf-8", errors="replace")





if __name__ == "__main__":
    import argparse
    from contextlib import ExitStack
    import tempfile
    import threading

    import psutil

    parser = argparse.ArgumentParser(description="Profile BPE training time and process-tree memory.")
    parser.add_argument(
        "input_path", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "owt_train.txt",
    )
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument(
        "--sample-mib", type=int,
        help="Train on the first N MiB in a temporary file; omit to use the full input.",
    )
    parser.add_argument(
        "--special-token", action="append",
        help="Repeat for multiple special tokens (default: <|endoftext|>).",
    )
    args = parser.parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    if not args.input_path.is_file():
        parser.error(f"Input file not found: {args.input_path}")
    if args.sample_mib is not None and args.sample_mib <= 0:
        parser.error("--sample-mib must be positive")
    if args.vocab_size < 256 + len(special_tokens):
        parser.error("--vocab-size must accommodate the 256 bytes and special tokens")

    with ExitStack() as cleanup:
        input_path = args.input_path
        if args.sample_mib is not None:
            sample_dir = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="bpe-profile-"))
            input_path = Path(sample_dir) / "sample.txt"
            remaining = args.sample_mib * 1024**2
            with args.input_path.open("rb") as source, input_path.open("wb") as target:
                while remaining > 0:
                    chunk = source.read(min(remaining, 1024**2))
                    if not chunk:
                        break
                    target.write(chunk)
                    remaining -= len(chunk)

        process = psutil.Process()
        stop_monitor = threading.Event()
        peak_rss = [0]
        memory_errors = []

        def sample_memory():
            try:
                total = 0
                for member in [process, *process.children(recursive=True)]:
                    try:
                        total += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss[0] = max(peak_rss[0], total)
            except (psutil.Error, OSError) as exc:
                memory_errors.append(str(exc))
                stop_monitor.set()

        def monitor_memory():
            while not stop_monitor.wait(0.05):
                sample_memory()

        input_bytes = input_path.stat().st_size
        print(f"Input: {args.input_path} ({input_bytes / 1024**2:,.2f} MiB used)", flush=True)
        print(f"Vocabulary target: {args.vocab_size:,}; training with existing BPE settings", flush=True)
        sample_memory()
        monitor = threading.Thread(target=monitor_memory, daemon=True)
        monitor.start()
        started = time.perf_counter()
        completed = False
        try:
            vocab, merges = BPE_train(input_path, args.vocab_size, special_tokens)
            completed = True
        finally:
            elapsed = time.perf_counter() - started
            stop_monitor.set()
            monitor.join()
            sample_memory()
            print(f"Elapsed training time: {elapsed:,.2f} s ({elapsed / 60:,.2f} min)", flush=True)
            if memory_errors:
                print(f"Peak memory unavailable/incomplete: {memory_errors[0]}", flush=True)
            else:
                print(f"Sampled peak RSS (parent + workers): {peak_rss[0] / 1024**3:,.3f} GiB", flush=True)
                print("RSS sampled every 50 ms; shared pages may be counted more than once.", flush=True)
            if completed:
                print(f"Throughput: {input_bytes / 1024**2 / elapsed:,.2f} MiB/s", flush=True)
                print(f"Produced {len(vocab):,} tokens and {len(merges):,} merges; no model files saved.", flush=True)
