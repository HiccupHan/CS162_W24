## Starter Code for CS162 HW1

Welcome to the coding part for the HW1!

IMPORTANT: To get the GloVE word embedding txt files follow the following steps:
1. Go to https://nlp.stanford.edu/projects/glove/
2. Download "Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download)" by clicking the link that states "glove.6B.zip"
3. You will need only glove.6B.50d.txt and glove.6B.200.txt so you can delete the other files
4. Move the glove glove word embedding into the same folder as your wordvec.py file


You could run the code with command like the following ones:

```
python wordvec.py --embeddings glove.6B.50d.txt --word dog --n 5

python wordvec.py --embeddings glove.6B.50d.txt --word king --minus man --plus woman --n 5
```
