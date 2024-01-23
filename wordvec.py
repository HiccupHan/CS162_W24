# CS162 HW1 Answer

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.
# Logging is in general a good practice to check the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# - It prints to standard error (stderr), not standard output (stdout) by
#   default. This means it won't interfere with the real output of your
#   program. 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
#
# In `parse_args`, we provided two command line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'. 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--embeddings", default="glove.6B.50d.txt", type=Path, 
                        help="Path to word embeddings file.")
    parser.add_argument("--word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)
    parser.add_argument("--n", type=int, default=5, 
                        help="Number of most similar words to be returned")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.
    """

    def __init__(self, file) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        self.word_emb_dict = {}
        ##################################################
        # Coding Task 1
        # Read in content of the embedding file and save 
        # the word embeddings to some data structure
        
        ##################################################

    def get_vector(self, word: String):
        """Return word vector of a specific word"""
        ##################################################
        # Coding Task 1
        # You might want to have a dedicated function
        # to return word vector of a specific word
        return []
        ##################################################

    def find_nearest_words(self, word, exclude_w, *, n = 5, plus: Optional[str] = None, minus: Optional[str] = None):
        """Find most similar words, in terms of embeddings, to a query."""
        # Params:
        #       word: the center word
        #       exclude_w: the words you want to exclude in the responses. It is a set in python.
        #       n: how many nearest result to be contained in the returned list
        #       plus: perform word analogy by adding embedding of the plus word to the center word
        #       minus: perform word analogy by substracting embedding of the minus word to the center word
        #
        # Return: 
        #       An iterable (e.g. a list) of up to n tuples of the form (word, score) where the k-th tuple 
        #       indicates the k-th most similar word to the input word and the similarity score
        #       If fewer than n words are available the function should return a shorter iterable
        #       Example:
        #           [(cat, 0.827517295965), (university, -0.190753135501)]
        #
        # The star above forces you to use `n`, `plus` and `minus` as
        # named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        #
        # Hint: You can use fast, batched computations
        # instead of looping over the rows.

        ##################################################
        # Coding Task 3 and 4
        return []
        ##################################################

def euclidean_distance(v1, v2):
    """Calculate the normalized euclidean distance between two vectors"""
    # Params:
    #       v1, v2: numpy arrays
    # Return:
    #       Euclidean distance, a number between 0 and sqrt(2)
    
    ##################################################
    # Coding Task 2
    # You can also use other way to compute norm, such as: np.sqrt(np.dot(v1, v1))
    # We set axis=0 so the norm is calculated along the 0 axis rather than the matrix norm
    # This function should work for both cases: 1) both v1 and v2 are 1-D array
    #                                        or 2) v1 is 1D array and v2 is 2D matrix for batch calculation
    
    return 0
    #################################################
    


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    # Load word vectors
    log.info("{}\nCoding task 1: loading word vectors".format("-"*10))
    lexicon = Lexicon(args.embeddings)
    log.info("Word vector of {} from {}: {}".format(args.word, args.embeddings, lexicon.get_vector(args.word)))
    
    log.info("{}\nCoding task 2: normalized euclidean distance".format("-"*10))
    euclidean_result = euclidean_distance(lexicon.get_vector(args.word), lexicon.get_vector("woman"))
    log.info("Euclidean distance between {} and {} is: {}".format(args.word, "woman", euclidean_result))
    euclidean_result = euclidean_distance(lexicon.get_vector(args.word), lexicon.get_vector("man"))
    log.info("Euclidean distance between {} and {} is: {}".format(args.word, "man", euclidean_result))

    log.info("{}\nCoding task 3: nearest neighbors".format("-"*10))
    sim = lexicon.find_nearest_words(args.word, {args.word}, n = args.n)
    log.info("Top {} similar words of {} are: {}".format(args.n, args.word, sim))

    log.info("{}\nCoding task 4: word analogy".format("-"*10))
    sim = lexicon.find_nearest_words(args.word, {args.word}, n = args.n, minus = args.minus, plus = args.plus)
    log.info("Top {} similar words of {} - {} + {} are: {}".format(args.n, args.word, args.minus, args.plus, sim))

if __name__ == '__main__':
    main()
