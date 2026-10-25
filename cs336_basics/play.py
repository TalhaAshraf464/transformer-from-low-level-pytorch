import regex as re
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time
import pickle
from pathlib import Path
from functools import partial




PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

test = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"



def split_on_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in special_tokens)
    pieces = re.split(f"({pattern})", text)
    return [piece for piece in pieces if piece]



def Pre_Tokenize_For_Vocab(text, special_tokens):
    """
    Takes text and a list of special tokens
    Returns a dictionary with tuples (representing different words/chunks from the text) 
    and their count in the text
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



def parallel_pretokenize(file_path, special_tokens, parallels=8):
    """"
    Takes the pre tokenizing code, splits its work across parallels number of
    cpu processes and runs pre tokenization on the file in file_path.
    """
    start_time = time.perf_counter()
    print("Parallel pre-tokenization has begun.")
    final_counts = Counter()

    with open(file_path, "rb") as f:
        list_of_texts = []
        num_processes = parallels
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            list_of_texts.append(f.read(end - start).decode("utf-8", errors="ignore"))

    with Pool(parallels) as p:
        arguments = [(text, special_tokens) for text in list_of_texts]
        list_of_counts = p.starmap(Pre_Tokenize_For_Vocab, arguments)

    for count in list_of_counts:
        final_counts.update(count)
    
    end_time = time.perf_counter()

    print("Pre-tokenization has concluded!")
    print(f"Time taken in parallel pretokenization is {end_time-start_time}")
    return final_counts



# def Parallel_Pretokenize_Memory_Saving(file_path, special_tokens, number_of_cpu_streams):
#     """
#     Takes a file path to a supposedly huge file. Streams a certain amount of lines into pre-tokeization at a time to save memory.
#     Pre-tokenization is split across parallel cpu processes to save time. 
#     """
#     start_time = time.perf_counter()
#     print("Parallel pre-tokenization has begun.")

#     Final_Count = Counter()

#     with Pool(number_of_cpu_streams) as p:  # create workers once
#         with open(file_path, "rb") as f:
#             batch = []

#             for line in f:
#                 batch.append(line)

#                 if len(batch) == 1000:
#                     batch_text = b"".join(batch)

#                     # Split this one batch across workers, then update Final_Count.
#                     boundaries = find_chunk_boundaries(
#                         batch_text,
#                         number_of_cpu_streams,
#                         b"<|endoftext|>",
#                     )

#                     list_of_texts = [
#                         batch_text[start:end].decode("utf-8")
#                         for start, end in zip(boundaries[:-1], boundaries[1:])
#                     ]

#                     arguments = [(text, special_tokens) for text in list_of_texts]
#                     list_of_counts = p.starmap(Pre_Tokenize_For_Vocab, arguments)

#                     for count in list_of_counts:
#                         Final_Count.update(count)

#                     batch.clear() # only clear after processing it
            
#     end_time = time.perf_counter()
#     print("Pre-tokenization has concluded!")
#     print(f"Time taken in parallel pretokenization is {end_time-start_time}")
#     return Final_Count




def max_from_dict(dictionary):
    return max(dictionary, key=lambda pair: (dictionary[pair], pair))


def create_vocab_mergelist(counts,vocabulary, N, merge_list = None):
    """"
    Takes in a counts dict and an initial vocabulary, Uses the BPE algorithm to find 
    N-len(vcabulary) more vocab words, returns a new vocabulary and a mergelist for the operations it did

    Returns a vocabulary of size N and a merge list which contians the history of merges for
    encoding purposes.

    vocabulary: dict of ID: bytestring
    merge)list: list of tuples of bytesrting pairs 
    """
    if merge_list is None:
        merge_list = []

    # Pair counts will have the number of times a pair shows up in the text.
    # Pair origins will have lists showing the chunks that pair comes from.
    pair_counts = {}
    pair_origins = {}
    for chunk in counts:
        for ind in range(len(chunk) - 1):
            pair = (chunk[ind], chunk[ind + 1])
            if pair not in pair_counts:
                pair_counts[pair] = counts[chunk]
                pair_origins[pair] = [chunk]
            else:
                pair_counts[pair] += counts[chunk]
                if chunk not in pair_origins[pair]:
                    pair_origins[pair].append(chunk)
    #################################

    
    def remove_chunk_pairs(chunk, count):
        pairs_in_chunk = set()
        for ind in range(len(chunk) - 1):
            old_pair = (chunk[ind], chunk[ind + 1])
            pair_counts[old_pair] -= count
            pairs_in_chunk.add(old_pair)

        for old_pair in pairs_in_chunk:
            pair_origins[old_pair].remove(chunk)
            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]
                del pair_origins[old_pair]

    def add_chunk_pairs(chunk, count):
        pairs_in_chunk = set()
        for ind in range(len(chunk) - 1):
            new_pair = (chunk[ind], chunk[ind + 1])
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count
            pairs_in_chunk.add(new_pair)

        for new_pair in pairs_in_chunk:
            if new_pair not in pair_origins:
                pair_origins[new_pair] = [chunk]
            elif chunk not in pair_origins[new_pair]:
                pair_origins[new_pair].append(chunk)

    def merge(counts, pair):
        list_of_origins = list(pair_origins[pair])
        for chunk in list_of_origins:
            ind = 0
            new_chunk = []
            while ind < len(chunk):
                check_pair = (chunk[ind], chunk[ind + 1]) if ind + 1 < len(chunk) else None

                if check_pair != pair:
                    new_chunk.append(chunk[ind])
                if check_pair == pair:
                    ind += 1
                    new_chunk.append(pair[0] + pair[1])
                ind += 1   

            val = counts[chunk]
            remove_chunk_pairs(chunk, val)
            del counts[chunk]
            new_chunk = tuple(new_chunk)
            counts[new_chunk] = counts.get(new_chunk, 0) + val
            add_chunk_pairs(new_chunk, val)

        return counts
 

    for x in range(N):
        print(f"{x/N *100} percent of the tokenization is done.")
        if not pair_counts:
            break
        pair = max_from_dict(pair_counts)
        vocabulary.append(pair[0]+pair[1])
        merge_list.append(pair)
        counts = merge(counts, pair)
    return vocabulary, merge_list



def BPE_Vocabulary(input_path, vocab_size, special_tokens):
    start_time = time.perf_counter()
    vocabulary = [token.encode("utf-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    N = vocab_size - 256 - len(special_tokens)
    counts = parallel_pretokenize(input_path, special_tokens)
    vocab_list , merge= create_vocab_mergelist(counts, vocabulary, N)

    end_time = time.perf_counter()
    print(f"Time taken in tokenization is {end_time-start_time}")

    return {i:vocab_list[i] for i in range(len(vocab_list))}, merge



#############################
#############################
#############################
#############################
#############################
#############################



class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else None
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.lookup_vocab_index = {token: id for token, id in self.vocab.items()}
        if self.special_tokens is not None:
            for stoken in self.special_tokens:
                stoken = stoken.encode("utf-8")
                if stoken not in self.lookup_vocab_index:
                    self.vocab[len(self.vocab)] = stoken


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            Vocab, Merges = pickle.load(f)
            return cls(Vocab, Merges, special_tokens)

    def encode(self, text):
        # Encode an input text into a sequence of token IDs. 

        #We start by turning the text into a list of chunks, each chunk a list of bytes
        strings = split_on_special_tokens(text, self.special_tokens)
        for str in strings:
            if self.special_tokens is not None and str not in self.special_tokens:    
                chunks = re.findall(PAT, str) #Returns a list of chunks we can vocab on
                byte_chunks = [list(bytes([b]) for b in chunk.encode("utf-8")) for chunk in chunks] #List of chunks where each chunk is broken down into a tuple of single bytes

        #Now we go through the chunks in the list, then mutate it with merges
        for chunk in byte_chunks:
            while len(chunk) > 1:
                pairs = [(chunk[i], chunk[i + 1])for i in range(len(chunk)-1)]

                valid_pairs = [pair for pair in pairs if pair in self.merge_ranks]

                if not valid_pairs:
                    break

                pair_to_merge = min(valid_pairs, key=self.merge_ranks.get)

                new_chunk = []
                i = 0

                while i < len(chunk):
                    if (i + 1 < len(chunk) and (chunk[i], chunk[i + 1]) == pair_to_merge):
                        new_chunk.append(chunk[i] + chunk[i + 1])
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1

                chunk = new_chunk

        #Now we go through again, and turn the chunk elements into token IDs
        IDs_list = []
        for chunk in byte_chunks:
            for token in chunk:
                IDs_list.append(self.lookup_vocab_index[token])
        return IDs_list

    def encode_iterable(self, iterable):
        out = []
        for line in iterable:
            Ids_line = self.encode(line)
        for token in Ids_line:
            yield token
        

    def decode(self, ids):
        byte_string = b"".join(self.vocab[id] for id in ids)
        return byte_string.decode("utf-8")


"""
Speed experiments:

"""



if __name__ == "__main__":

    # vocab, merges = BPE_Vocabulary(ff, 10000, ["<|endoftext|>"])
    # trained_data_path = Path(__file__).parent / "trained_data"
    # trained_data_path.mkdir(exist_ok=True)
    # with open(trained_data_path / "bpe_tokenizer_10k_owt.pkl", "wb") as f:
    #     pickle.dump((vocab, merges), f)

    vocab, merges = BPE_Vocabulary("assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 32000, ["<|endoftext|>"])
    trained_data_path = Path(__file__).parent / "trained_data"
    trained_data_path.mkdir(exist_ok=True)
    with open(trained_data_path / "bpe_tokenizer_32k_tinystories.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)

    # vocab, merges = BPE_Vocabulary("assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    # trained_data_path = Path(__file__).parent / "trained_data"
    # trained_data_path.mkdir(exist_ok=True)
    # with open(trained_data_path / "bpe_tokenizer.pkl", "wb") as f:
    #     pickle.dump((vocab, merges), f)
    pass