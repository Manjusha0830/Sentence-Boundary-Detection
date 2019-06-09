NLP - SENTENCE BOUNDARY DETECTION

Python program SBD.py that detects sentence boundaries in text. Specifically, to predict if a period (.) is the end of a sentence or not. Train and test your program on subsets of the Treebank dataset, consisting of documents drawn from various sources, which have been manually tokenized and annotated with sentence boundaries.

Assumptions:
●	program should only focus on occurrences of the period (.), thus assuming that no other punctuation marks will end a sentence. In other words, in your program, you should not handle question marks (?) or exclamation signs (!).
●	program should only handle periods that end a word. That is, you should ignore those periods that are embedded in a word, e.g., Calif.-based or 27.4
●	program will only make use of the EOS (End of Sentence) and NEOS (Not End of Sentence) labels, and ignore all the TOK labels.

Programming guidelines:
program should perform the following steps:
➢	Identify all the period occurrences where the period ends a word, i.e., sequences of the form “L. R” Each of these periods is labeled as either EOS or NEOS.
➢	For each such period occurrence:
➔ Extract the set of five core features described in class, namely:
1.	Word to the left of “.” (L) (values: English vocab)
2.	Word to the right of “.” (R) (values: English vocab)
3.	Length of L < 3 (values:  binary)
4.	Is L capitalized (values: binary)
5.	Is R capitalized (values: binary)
➔ Extract three additional features of your own choosing.
6.checks if Left word contains any "."
7. checks if Right word contains any "."
8. checks if Left word is numeric

 
➢	The two steps above will create, for both the training and the test dataset, a collection of feature vectors. Each feature vector corresponds to one period instance, consists of eight features, and is assigned an EOS or a NEOS label.
➢	Using the sklearn library, train a Decision Tree classifier. Use the feature vectors obtained from the examples in SBD.train to train the classifier, and apply the resulting model to predict the labels for the feature vectors obtained from the examples in SBD.test.
➢	Compare the labels predicted by the classifier for the feature vectors obtained from SBD.test against the provided (gold-standard) labels and calculate the accuracy of your system.

The SBD.py program should be run using a command like this:
% python SBD.py SBD.train SBD.test

The program should produce at the standard output the accuracy of the system, as a percentage. It should also generate a file called SBD.test.out, which includes the first two columns from the input file SBD.test, along with the label EOS or NEOS predicted by the system for each period occurrence in the test data. The version of the program that you will submit should include all eight features mentioned before (core + your own).

