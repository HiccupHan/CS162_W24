import unittest
import math
import language_model

class TestLanguageModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.train_data, self.dev_data = language_model.load_data()

    def test_ngram(self):
        lm = language_model.LanguageModel(self.train_data, N=1, verbose=False)
        freq_ngrams = sorted(lm.train_ngram_freq.items(), key=lambda x: x[1], reverse=True)[:1] 
        self.assertEqual(freq_ngrams, [(('the',), 68100)])
        lm = language_model.LanguageModel(self.train_data, N=2, verbose=False)
        freq_ngrams = sorted(lm.train_ngram_freq.items(), key=lambda x: x[1], reverse=True)[:1] 
        self.assertEqual(freq_ngrams, [(('<bos>', 'the'), 10893)])

    def test_basic_perplexity(self):
        lm = language_model.LanguageModel(self.train_data, N=1, verbose=False)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 1231.4116015165341)
        self.assertTrue(math.isnan(lm.corpus_perplexity(self.dev_data)))
        lm = language_model.LanguageModel(self.train_data, N=2, verbose=False)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 62.52893276577851) 
        self.assertTrue(math.isnan(lm.corpus_perplexity(self.dev_data)))

    def test_smooth_perplexity(self):
        lm = language_model.LanguageModel(self.train_data, N=1, add_k = 1, verbose=False)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 1236.977852779304)
        self.assertAlmostEqual(lm.corpus_perplexity(self.dev_data), 1263.0795439418666)
        lm = language_model.LanguageModel(self.train_data, N=2, add_k = 1, verbose=False)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 2272.8805883822974)
        self.assertAlmostEqual(lm.corpus_perplexity(self.dev_data), 2818.1539056724932)

    def test_backoff(self):
        lm = language_model.LanguageModel(self.train_data, N=3, backoff=True, lambdas=[0.34, 0.33, 0.33], verbose=False)
        self.assertAlmostEqual(lm.corpus_perplexity(self.train_data), 15.845769774401393)
        self.assertAlmostEqual(lm.corpus_perplexity(self.dev_data), 136.990977837316)
        
    def test_generate_sent(self):
        lm = language_model.LanguageModel(self.train_data, N=1, verbose = False)
        self.assertEqual(lm.greedy_search(), ' '.join(['the']*50))
        lm = language_model.LanguageModel(self.train_data, N=2, verbose=False)
        self.assertEqual(lm.greedy_search(), '<bos> the company said <eos>')
        

if __name__ == '__main__':
    unittest.main()
