# Machine-Translation-NLP_POS-Tag-Analyser
Given a training corpus (train.txt) containing POS tagging for English sentences to learn a POS tagging model and then give precision, recall and F1 Score for each POS tag on the test corpus (test1.txt).

<br/>
Tasks: <br/>
 Calculate the bigram and lexical probabilities <br/>
 Viterbi algorithm is used for POS tagging<br/><br/>
 The performance measures (precision, recall and F1 Score) made to be reported in a text file (output.txt). <br/><br/>
Format of train.txt, test1.txt and test2.txt files is:<br/>
 Each line represents a sentence<br/>
 '#' symbol: separates a word from its POS tag<br/>
 '@' symbol: separates one word from another<br/><br/>
 Example:<br/>
Corpus: John#NOUN@entered#VERB@the#DET@room#NOUN@from#ADP@the#DET@hallway#NOUN@to#ADP@the#DET@kitchen#NOUN@.#.<br/>
Sentence: John entered the room from the hallway to the kitchen.<br/>
POS tags: NOUN VERB DET NOUN ADP DET NOUN ADP DET NOUN .<br/>
