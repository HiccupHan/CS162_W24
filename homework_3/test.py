import unittest
import pos_tagger

class TestLanguageModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.train_data, self.dev_data = pos_tagger.load_data()
        self.pos_tagger = pos_tagger.POSTagger(self.train_data)

    def test_tran_prob(self):
        self.assertAlmostEqual(self.pos_tagger.tran_prob[('PRON', 'VERB')], 0.7060998151571165)
        self.assertAlmostEqual(self.pos_tagger.tran_prob[('VERB', 'ADB')],  0)

    def test_emis_prob(self):
        self.assertAlmostEqual(self.pos_tagger.emis_prob[('DET', 'a')], 0.1594847480384341)
        self.assertAlmostEqual(self.pos_tagger.emis_prob[('NOUN', 'love')], 0.0005204755064226678)

    def test_init_prob(self):
        self.assertAlmostEqual(self.pos_tagger.init_prob('PRON'), 0.1607)
        self.assertAlmostEqual(self.pos_tagger.init_prob('VERB'), 0.048)

    def test_viterbi(self):
        self.assertEqual(self.pos_tagger.viterbi(["Remove", "all", "the", "loose", "spacing", "bars", "."]), ["VERB", "PRT", "DET", "ADJ", "VERB", "NOUN", "."])
        self.assertEqual(self.pos_tagger.viterbi(["Aligning", "all","the", "teeth", "may", "take", "a", "year", "more", "."]), ['VERB', 'PRT', 'DET', 'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'ADV', '.'])
        

if __name__ == '__main__':
    unittest.main()
