from collections import Counter
import argparse
import math

def load_data():
    """
    Loading training and dev data. We append an eos token at the end of each sentence.
    """
    train_path = 'data/train.txt' # the data paths are hard-coded 
    dev_path  = 'data/dev.txt'
    
    with open(train_path, 'r') as f:
        train_data = [f'{l.strip()} <eos>' for l in f.readlines()]
    with open(dev_path, 'r') as f:
        dev_data = [f'{l.strip()} <eos>' for l in f.readlines()]
    return train_data, dev_data

class LanguageModel(object):
    def __init__(self, train_data, N=1, add_k=0.0, backoff=False, lambdas=[0.2, 0.4, 0.4], verbose=False):
        """
        Args:
            train_data (list of str): list of sentences comprising the training corpus.
            N (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
            backoff (bool): whether to use backoff smoothing.
            verbose (bool): whether output information for debug.
        """
        self.N = N
        # We create a vocabulary that is a Python Counter object of (word: frequency) key-value pairs.
        self.vocab = Counter( [words for sent in train_data for words in sent.split()])
        bos_counter = {"<bos>":len(train_data)}
        self.vocab.update(bos_counter)
        self.vocab_size = len(self.vocab)
        self.word_cnt = sum(self.vocab.values()) # word count can be used for unigram models
        self.add_k = add_k
        self.backoff = backoff
        self.lambdas = lambdas
        self.verbose = verbose
        if self.verbose:
            print(f'Vocabulary size: {self.vocab_size}')

        # We insert bos tokens in front of each sentence.
        # For N-gram LMs, we insert (N-1) bos token(s).
        # In this question, we do not insert any bos tokens for unigram models.
        self.bos = ' '.join(['<bos>']*(self.N-1))
        self.train_data = [f'{self.bos} {sent}' for sent in train_data]

        # We then count the frequency of N-grams.
        self.train_ngram_freq = self._get_ngrams_freq(self.train_data, N)
        self.train_n_minus_one_gram_freq = self._get_ngrams_freq(self.train_data, N-1) if N > 1 else None
       
    def _get_ngrams(self, sent, n):
        """
        Given a text sentence and the argument n, we convert it to a list of n-gram tuples.

        Args:
            sent (str): input text sentence.
            n (int): the order of n-grams to return (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngrams (list of tuple): list of n-gram tuples

        Example 1:
            Input: sent="<bos> I like NLP <eos>", n=2
            Output: [("<bos>", "I"), ("I", "like"), ("like", "NLP"), ("NLP", "<eos>")]
        Example 2:
            Input: sent="<bos> I like NLP <eos>", n=1
            Output: [("<bos>",), ("I",), ("like",), ("NLP",), ("<eos>",)]
        """
        ##################################################
        # Coding Task 1
        # Read in a sentence and the n and construct 
        # an n-gram tuple         
        ##################################################
        words = sent.split()
        ngrams = []
        for i in range(len(words)-n+1):
            # TODO: construct a ngram tuple with the first element being {words[i]}
            
            pass
        
        return ngrams

    def _get_ngrams_freq(self, corpus, n):
        """
        Given a training corpus, count the frequency of each n-gram.

        Args:
            corpus (list of str): list of sentences comprising the training corpus with <bos> inserted.
            n (int): the order of n-grams to count (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngram_freq (Counter): Python Counter object of (ngram (tuple), frequency (int)) key-value pairs
        """
        corpus_ngrams = []
        for sent in corpus:
            sent_ngrams = self._get_ngrams(sent, n)
            corpus_ngrams += sent_ngrams
        ngram_freq = Counter(corpus_ngrams)

        if self.verbose and n == self.N:
            print(f'Top 5 most frequent {n}-grams in the training data along with their frequency:')
            print(sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:5])
        return ngram_freq

    
    def sent_perplexity(self, sent):
        """
        Calculate the perplexity of the model against a given sentence
        
        Args:
            sent (str): a text sentence without <bos>.
        Returns:
            perplexity (float): the perplexity of the model as a float.
            log_prob (float): the log probability of this sentence.
            cnt_ngrams (int): the total number of n-grams in this sentence.
        """
        ##################################################
        # Coding Task 2
        # Implement sentence level perplexity using 
        # Jelinek-Mercer Backoff Smoothing; if backoff 
        # smoothing is turned off, implement a simple 
        # add-one smoothing.       
        ##################################################
        
        # We first insert bos tokens to the sentence and get its ngrams.
        sent = f'{self.bos} {sent}'
        sent_ngrams = self._get_ngrams(sent, self.N)

        log_prob = 0.0
        # We then iterate over each ngram.
        for ngram in sent_ngrams:
            
            if self.backoff == True and self.N!=3:
                raise Exception("You have entered wrong combination of CLI arguments. Check again!")
            
            # Case 1: unigram model (Add k smoothing where required)
            if self.N == 1:
                # TODO: 
                # 1. unigram exists and k == 0
                # 2. unigram does not exist or k > 0
                
                pass
            
            # case 2: n gram model for n > 1 (Add k smoothing where required)
            elif self.backoff == False:
                # TODO: (you might require n minus gram here)
                # 1. when ngram exists and add k = 0
                # 2. when ngram exists and add k > 0
                # 3. when ngram does not exists, add k > 0 
                
                pass
            
            # Case 3: special case of simple linear interpolation for a trigram (No add k smoothing here)
            elif self.backoff == True:  
                # TODO: 
                # p(avg) = lambda1 * p(trigram) + lambda1 * p(trigram) + lambda1 * p(trigram)
                # you need to find appropriate value of lambda such that p(avg) is highest
                # or log prob is less negative
                # deal with 0/0 division as no add one smoothing used
                
                pass

        cnt_ngrams = len(sent_ngrams)
        # TODO: compute sentence-level perplexity
        
        perplexity = None
        return perplexity, log_prob, cnt_ngrams


    def corpus_perplexity(self, corpus):
        """
        Calculate the perplexity of the model against a corpus.
        Here, we iterate over the statistics of each individual sentence,
        based on which we get the final corpus-level perplexity.
            
        Args:
            corpus (list of str): list of sentences.
        Returns:
            perplexity (float): the perplexity of the model as a float.
        """
        ##################################################
        # Coding Task 3
        # Implement corpus level perplexity using 
        # sentence perplexity      
        ##################################################
        corpus_log_prob = 0
        corpus_cnt_ngrams = 0
        for sent in corpus:
            _, sent_log_prob, sent_cnt_ngrams = self.sent_perplexity(sent)
            corpus_log_prob += sent_log_prob
            corpus_cnt_ngrams += sent_cnt_ngrams
       
        # TODO: compute corpus-level perplexity. The equation should be almost the same to 
        #       sentence-level perplexity.
            
        perplexity = None
        return perplexity
        
    def greedy_search(self, max_steps=50):
        """
        Generate the most probable sentence by doing greedy search on the model.
            
        Args:
            max_steps (int): the maximum length of the generated sentence (default: 50).
        Returns:
            a string of the generated sentence.
        """
        ##################################################
        # Coding Task 4
        # Implement greedy serach which generates the most 
        # probable sentence for that n-gram model   
        ##################################################
        # The sentence first start with the bos token(s).
        words = self.bos.split()

        while len(words) < max_steps:
            # At each step, we generate one word based on n-gram frequency.
            next_word = None
            next_max_freq = -1
            for word in self.vocab:
                # We iterate over the vocabulary and select the most probable word.
                # For each candidate word, we construct its corresponding n-gram.
                if self.N == 1:
                    ngram = tuple([word])
                else:
                    ngram = tuple(words[-self.N+1:] + [word])
                # TODO: given the n-gram, we retrieve its frequency in the training data.
                #       We update {next_word} and {next_max_freq} under certain conditions.
                pass
            if next_word is None:
                break
            words.append(next_word) 
            if next_word == '<eos>':
                break

        return ' '.join(words)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True,
            help='Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)')
    parser.add_argument('--k', type=float, default=0.0,
            help='Parameter for add-k, smoothing (default is 0.0 (False) -- set to True for add-k smoothing)')
   
    parser.add_argument('--backoff', action="store_true", default=False,
            help='Parameter for backoff smoothing (default is False (disabled) -- set to True for deleted interpolation smoothing, specifically, Jelinek-Mercer Backoff Smoothing)')
    parser.add_argument('--lambdas', nargs='+', type=float, default=[0.2, 0.4, 0.4],
            help='lambdas for simple linear interpolation, in order, lambda3 lambda2 lambda1, \
                Eg. --lambdas 0.2 0.4 0.4 means lambda3=0.2, lambda2=0.4 and lambda1=0.4')

    parser.add_argument('--verbose', type=bool, default=True,
            help='Will print information that is helpful for debug if set to True')
    
    args = parser.parse_args()

    # Loading data and display samples
    train_data, dev_data = load_data()
    if args.verbose:
        print('First 3 elements in the loaded training data:')
        print(train_data[:3])
        print('First 3 elements in the loaded dev data:')
        print(dev_data[:3])

    # Building a language model instance
    lm = LanguageModel(train_data, N = args.N, add_k=args.k, backoff=args.backoff, lambdas=args.lambdas, verbose=args.verbose)

    # Computing the perplexity of this language model on the training and dev data
    print(f'Perplexity on the training data: \n{lm.corpus_perplexity(train_data)}')
    print(f'Perplexity on the dev data: \n{lm.corpus_perplexity(dev_data)}')

    # Generating the most probable sentence using greedy search
    print(f'Generating the most probable sentence using greedy search: \n{lm.greedy_search()}')
