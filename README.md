# awesome-text-summarization

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources dedicated to text summarization

## Contents

### Corpus

1. [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) contains 51 articles. Each article is about a product’s feature, like iPod’s Battery Life, etc. and is a collection of reviews by customers who purchased that product. Each article in the dataset has 5 manually written “gold” summaries. Usually the 5 gold summaries are different but they can also be the same text repeated 5 times.
2. [DUC](http://duc.nist.gov/)
3. [English Gigaword](https://catalog.ldc.upenn.edu/LDC2003T05): English Gigaword was produced by Linguistic Data Consortium (LDC).
6. [Large Scale Chinese Short Text Summarization Dataset（LCSTS）](http://icrc.hitsz.edu.cn/Article/show/139.html): This corpus is constructed from the Chinese microblogging website SinaWeibo. It consists of over 2 million real Chinese short texts with short summaries given by the writer of each text.
7. Ziqiang Cao, Chengyao Chen, Wenjie Li, Sujian Li, Furu Wei, Ming Zhou. [TGSum: Build Tweet Guided Multi-Document Summarization Dataset](https://arxiv.org/abs/1511.08417v1). arXiv:1511.08417, 2015.
9. [scisumm-corpus](https://github.com/WING-NUS/scisumm-corpus) contains a release of the scientific document summarization corpus and annotations from the WING NUS group.
12. Avinesh P.V.S., Maxime Peyrard, Christian M. Meyer. [Live Blog Corpus for Summarization](https://arxiv.org/abs/1802.09884v1). arXiv:1802.09884, 2018.
13. Alexander R. Fabbri, Irene Li, Prawat Trairatvorakul, Yijiao He, Wei Tai Ting, Robert Tung, Caitlin Westerfield, Dragomir R. Radev.[TutorialBank: A Manually-Collected Corpus for Prerequisite Chains, Survey Extraction and Resource Recommendation](https://arxiv.org/abs/1805.04617). arXiv:1805.04617, 2018. The source code is [TutorialBank](https://github.com/Yale-LILY/TutorialBank). All the datasets could be found through the [search engine](http://tangra.cs.yale.edu/newaan/). The blog [TutorialBank: Learning NLP Made Easier](https://alex-fabbri.github.io/TutorialBank/) is an excellent user guide with step by step instructions on how to use the search engine.


### Text Summarization Software

1. [sumeval](https://github.com/chakki-works/sumeval) implemented in Python is a well tested & multi-language evaluation framework for text summarization.
2. [sumy](https://github.com/miso-belica/sumy) is a simple library and command line utility for extracting summary from HTML pages or plain texts. The package also contains simple evaluation framework for text summaries. Implemented summarization methods are *Luhn*, *Edmundson*, *LSA*, *LexRank*, *TextRank*, *SumBasic* and *KL-Sum*.
3. [TextRank4ZH](https://github.com/letiantian/TextRank4ZH) implements the *TextRank* algorithm to extract key words/phrases and text summarization
in Chinese. It is written in Python.
4. [snownlp](https://github.com/isnowfy/snownlp) is python library for processing Chinese text.
5. [PKUSUMSUM](https://github.com/PKULCWM/PKUSUMSUM) is an integrated toolkit for automatic document summarization. It supports single-document, multi-document and topic-focused multi-document summarizations, and a variety of summarization methods have been implemented in the toolkit. It supports Western languages (e.g. English) and Chinese language.
6. [fnlp](https://github.com/FudanNLP/fnlp) is a toolkit for Chinese natural language processing.

### Word Representation

1. G. E. Hinton, J. L, McClelland, and D. E. Rumelhart. [Distributed representations](https://web.stanford.edu/~jlmcc/papers/PDP/Chapter3.pdf). In D. E. Rumelhart and J. L. McClelland, Parallel Distributed Processing: Explorations in the Microstructure of Cognition. Volume 1: Foundations, MIT Press, Cambridge, MA. 1986. The related slides are [here](http://www.cs.toronto.edu/~bonner/courses/2014s/csc321/lectures/lec5.pdf) or [here](http://www.cs.toronto.edu/~bonner/courses/2016s/csc321/lectures/extra/coarse.pdf).
2. [N-Grams](https://lagunita.stanford.edu/c4x/Engineering/CS-224N/asset/slp4.pdf)
3. Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin. [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). 2003.
   - They proposed to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences.
1. Christopher Olah. [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/). This post reviews some extremely remarkable results in applying deep neural networks to NLP, where the representation perspective of deep learning is a powerful view that seems to answer why deep neural networks are so effective. 
4. Levy, Omer, and Yoav Goldberg. [Neural word embedding as implicit matrix factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf). NIPS. 2014.
5. [Sanjeev Arora](http://www.offconvex.org/)'s a series of blogs/papers about word embeddings:
   * The blog [Semantic Word Embeddings](http://www.offconvex.org/2015/12/12/word-embeddings-1/) is a very good overview about word embedding.
   * The blog [Word Embeddings: Explaining their properties](http://www.offconvex.org/2016/02/14/word-embeddings-2/) introduces the main result about [RAND-WALK: A Latent Variable Model Approach to Word Embeddings](https://arxiv.org/abs/1502.03520), which answers three interesting questions: 1. Why do low-dimensional embeddings capture huge statistical information? 2. Why do low dimensional embeddings work better than high-dimensional ones? 3. Why do Semantic Relations correspond to Directions?
   * The blog [Linear algebraic structure of word meanings](http://www.offconvex.org/2016/07/10/embeddingspolysemy/) introduces the main result about [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://arxiv.org/abs/1601.03764), which shows that word senses are easily accessible in many current word embeddings.
1. [Word2Vec Resources](http://mccormickml.com/2016/04/27/word2vec-resources/): This is a post with links to and descriptions of word2vec tutorials, papers, and implementations.
2. [Word embeddings: how to transform text into numbers](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/)
3. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus.
1. Li, Yitan, et al. [Word embedding revisited: A new representation learning and explicit matrix factorization perspective](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.719.9973&rep=rep1&type=pdf). IJCAI. 2015.
1. O. Levy, Y. Goldberg, and I. Dagan. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016). Trans. Assoc. Comput. Linguist., 2015.
2. Eric Nalisnick, Sachin Ravi. [Learning the Dimensionality of Word Embeddings](https://arxiv.org/abs/1511.05392v3). arXiv:1511.05392, 2015.
   * They describe a method for learning word embeddings with data-dependent dimensionality. Their Stochastic Dimensionality Skip-Gram (SD-SG) and Stochastic Dimensionality Continuous Bag-of-Words (SD-CBOW) are nonparametric analogs of Mikolov et al.'s (2013) well-known 'word2vec' model.
1. William L. Hamilton, Jure Leskovec, Dan Jurafsky. [Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change](https://aclanthology.info/pdf/P/P16/P16-1141.pdf).
   * Hamilton et al. model changes in word meaning by fitting word embeddings on consecutive corpora of historical language. They compare several ways of quantifying meaning (co-occurrence vectors weighted by PPMI, SVD embeddings and word2vec embeddings), and align historical embeddings from different corpora by finding the optimal rotational alignment that preserves the cosine similarities as much as possible.
1. Zijun Yao, Yifan Sun, Weicong Ding, Nikhil Rao, Hui Xiong. [Dynamic Word Embeddings for Evolving Semantic Discovery](https://arxiv.org/abs/1703.00607). arXiv:1703.00607v2, International Conference on Web Search and Data Mining (WSDM 2018).
1. Yang, Wei  and  Lu, Wei  and  Zheng, Vincent. [A Simple Regularization-based Algorithm for Learning Cross-Domain Word Embeddings](http://www.aclweb.org/anthology/D/D17/D17-1312.pdf). ACL, 2017. The source code in C is [cross_domain_embedding](https://github.com/Victor0118/cross_domain_embedding).
   - This paper presents a simple yet effective method for learning word embeddings based on text from different domains.
2. Sebastian Ruder. [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/)
2. Bryan McCann, James Bradbury, Caiming Xiong and Richard Socher. [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107). For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors). 
   * A Keras/TensorFlow implementation of the MT-LSTM/CoVe is [CoVe](https://github.com/rgsachin/CoVe).
   * A PyTorch implementation of the MT-LSTM/CoVe is [cove](https://github.com/salesforce/cove).
2. Maria Pelevina, Nikolay Arefyev, Chris Biemann, Alexander Panchenko. [Making Sense of Word Embeddings](https://arxiv.org/abs/1708.03390). arXiv:1708.03390, 2017. The source code written in Python is [sensegram](https://github.com/tudarmstadt-lt/sensegram).
   - Making sense embedding out of word embeddings using graph-based word sense induction.
1. Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606). arXiv:1607.04606. 2017. The souce code in C++11 is [fastText](https://github.com/facebookresearch/fastText), which is a library for efficient learning of word representations and sentence classification.
1. Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer and Herv{\'e} J{\'e}gou. [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087). arXiv:1710.04087, 2017. The source code in Python is [MUSE](https://github.com/facebookresearch/MUSE), which is a library for multilingual unsupervised or supervised word embeddings.
1. Tomas Mikolov, Edouard Grave, Piotr Bojanowski, Christian Puhrsch and Armand Joulin. [Advances in Pre-Training Distributed Word Representations](https://arxiv.org/abs/1712.09405). arXiv:1712.09405, 2017.
2. Gabriel Grand, Idan Asher Blank, Francisco Pereira, Evelina Fedorenko. [Semantic projection: recovering human knowledge of multiple, distinct object features from word embeddings](https://arxiv.org/abs/1802.01241v2). arXiv:1802.01241, 2018.
   * Could context-dependent relationships be recovered from word embeddings? To address this issue, they introduce a powerful, domain-general solution: "semantic projection" of word-vectors onto lines that represent various object features, like size (the line extending from the word "small" to "big"), intelligence (from "dumb" to "smart"), or danger (from "safe" to "dangerous").
1. Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365). arXiv:1802.05365, NAACL 2018. The code is [ELMo](https://allennlp.org/elmo).
2. Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, Tomas Mikolov. [Learning Word Vectors for 157 Languages](https://arxiv.org/abs/1802.06893v2). arXiv:1802.06893v2, Proceedings of LREC, 2018.
3. Douwe Kiela, Changhan Wang and Kyunghyun Cho. [Context-Attentive Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983). arXiv:1804.07983, 2018. 
   * While one of the first steps in many NLP systems is selecting what embeddings to use, they argue that such a step is better left for neural networks to figure out by themselves. To that end, they introduce a novel, straightforward yet highly effective method for combining multiple types of word embeddings in a single model, leading to state-of-the-art performance within the same model class on a variety of tasks.
4. Laura Wendlandt, Jonathan K. Kummerfeld, Rada Mihalcea. [Factors Influencing the Surprising Instability of Word Embeddings](https://arxiv.org/abs/1804.09692v1). arXiv:1804.09692, NAACL HLT 2018.
   * They provide empirical evidence for how various factors contribute to the stability of word embeddings, and analyze the effects of stability on downstream tasks.

### Sentence Representation

1. Kalchbrenner, Nal, Edward Grefenstette, and Phil Blunsom. [A convolutional neural network for modelling sentences](http://arxiv.org/abs/1404.2188). arXiv:1404.2188, 2014.
2. Quoc Le and Tomas Mikolov. [Distributed representations of sentences and documents](http://arxiv.org/abs/1405.4053). arXiv:1405.4053, 2014.
3. Yoon Kim. [Convolutional neural networks for sentence classification](http://arxiv.org/abs/1408.5882). arXiv:1408.5882, EMNLP 2014.
1. Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun and Sanja Fidler. [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726). arXiv:1506.06726, 2015. The source code in Python is [skip-thoughts](https://github.com/ryankiros/skip-thoughts). The TensorFlow implementation of *Skip-Thought Vectors* is [skip_thoughts](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)
2. John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu. [Towards Universal Paraphrastic Sentence Embeddings](https://arxiv.org/abs/1511.08198). arXiv:1511.08198, ICLR 2016. The source code written in Python is [iclr2016](https://github.com/jwieting/iclr2016).
2. Zhe Gan, Yunchen Pu, Ricardo Henao, Chunyuan Li, Xiaodong He, Lawrence Carin. [Learning Generic Sentence Representations Using Convolutional Neural Networks](https://arxiv.org/abs/1611.07897). arXiv:1611.07897, EMNLP 2017. The training code written in Python is [ConvSent](https://github.com/zhegan27/ConvSent).
3. Matteo Pagliardini, Prakhar Gupta, Martin Jaggi. [Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/abs/1703.02507v2). arXiv:1703.02507, NAACL 2018. The source code in Python is [sent2vec](https://github.com/epfml/sent2vec). 
1. Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, Jason Weston. [StarSpace: Embed All The Things](https://arxiv.org/abs/1709.03856). arXiv:1709.03856, 2017. The source code in C++11 is [StarSpace](https://github.com/facebookresearch/Starspace/).
2. Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, Antoine Bordes. [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364). arXiv:1705.02364v4, EMNLP 2017. The source code in Python is [InferSent](https://github.com/facebookresearch/InferSent).
3. Sanjeev Arora, Yingyu Liang, Tengyu Ma. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx). ICLR 2017. The source code written in Python is [SIF](https://github.com/PrincetonML/SIF). [SIF_mini_demo](https://github.com/PrincetonML/SIF_mini_demo) is a minimum example for the sentence embedding algorithm.
1. Yixin Nie, Mohit Bansal. [Shortcut-Stacked Sentence Encoders for Multi-Domain Inference](https://arxiv.org/abs/1708.02312). arXiv:1708.02312, EMNLP 2017. The source code in Python is [multiNLI_encoder](https://github.com/easonnie/multiNLI_encoder). The new repo [ResEncoder]( https://github.com/easonnie/ResEncoder) is for Residual-connected sentence encoder for NLI.
2. Lajanugen Logeswaran, Honglak Lee. [An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893). arXiv:1803.02893, ICLR 2018. The open review comments are listed [here](https://openreview.net/forum?id=rJvJXZb0W).
3. Eric Zelikman. [Context is Everything: Finding Meaning Statistically in Semantic Spaces](https://arxiv.org/abs/1803.08493). arXiv:1803.08493, 2018.
1. Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil. [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175). arXiv:1803.11175v2, 2018.

### Extractive Text Summarization

1. H. P. Luhn. [The automatic creation of literature abstracts](http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf). IBM Journal of Research and Development, 1958. Luhn's method is as follows:
   1. Ignore Stopwords: Common words (known as stopwords) are ignored.
   2. Determine Top Words: The most often occuring words in the document are counted up.
   3. Select Top Words: A small number of the top words are selected to be used for scoring.
   4. Select Top Sentences: Sentences are scored according to how many of the top words they contain. The top four sentences are selected for the summary.
2. H. P. Edmundson. [New Methods in Automatic Extracting](http://courses.ischool.berkeley.edu/i256/f06/papers/edmonson69.pdf). Journal of the Association for Computing Machinery, 1969.
3. David M. Blei, Andrew Y. Ng and Michael I. Jordan. [Latent Dirichlet Allocation](http://ai.stanford.edu/~ang/papers/jair03-lda.pdf). Journal of Machine Learning Research, 2003. The source code in Python is [sklearn.decomposition.LatentDirichletAllocation](http://scikit-learn.org/dev/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html). Reimplement Luhn's algorithm, but with topics instead of words and applied to several documents instead of one.
   1. Train LDA on all products of a certain type (e.g. all the books)
   2. Treat all the reviews of a particular product as one document, and infer their topic distribution
   3. Infer the topic distribution for each sentence
   4. For each topic that dominates the reviews of a product, pick some sentences that are themselves dominated by that topic.
4. David M. Blei. [Probabilistic Topic Models](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf). Communications of the ACM, 2012.
5. Rada Mihalcea and Paul Tarau. [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf). ACL, 2004. The source code in Python is [pytextrank](https://github.com/ceteri/pytextrank). `pytextrank` works in four stages, each feeding its output to the next:
   - Part-of-Speech Tagging and lemmatization are performed for every sentence in the document.
   - Key phrases are extracted along with their counts, and are normalized.
   - Calculates a score for each sentence by approximating jaccard distance between the sentence and key phrases.
   - Summarizes the document based on most significant sentences and key phrases.
6. Federico Barrios, Federico López, Luis Argerich and Rosa Wachenchauzer. [Variations of the Similarity Function of TextRank for Automated Summarization](https://arxiv.org/abs/1602.03606). arXiv:1602.03606, 2016. The source code in Python is [gensim.summarization](http://radimrehurek.com/gensim/). Gensim's summarization only works for English for now, because the text is pre-processed so that stop words are removed and the words are stemmed, and these processes are language-dependent. TextRank works as follows:
   - Pre-process the text: remove stop words and stem the remaining words.
   - Create a graph where vertices are sentences.
   - Connect every sentence to every other sentence by an edge. The weight of the edge is how similar the two sentences are.
   - Run the PageRank algorithm on the graph.
   - Pick the vertices(sentences) with the highest PageRank score.
7. [TextTeaser](https://github.com/MojoJolo/textteaser) uses basic summarization features and build from it. Those features are:
   - Title feature is used to score the sentence with the regards to the title. It is calculated as the count of words which are common to title of the document and sentence.
   - Sentence length is scored depends on how many words are in the sentence. TextTeaser defined a constant “ideal” (with value 20), which represents the ideal length of the summary, in terms of number of words. Sentence length is calculated as a normalized distance from this value.
   - Sentence position is where the sentence is located. I learned that introduction and conclusion will have higher score for this feature.
   - Keyword frequency is just the frequency of the words used in the whole text in the bag-of-words model (after removing stop words).
8. Güneş Erkan and Dragomir R. Radev. [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html). 2004.
   - LexRank uses IDF-modified Cosine as the similarity measure between two sentences. This similarity is used as weight of the graph edge between two sentences. LexRank also incorporates an intelligent post-processing step which makes sure that top sentences chosen for the summary are not too similar to each other.
9. [Latent Semantic Analysis(LSA) Tutorial](https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/).
10. Josef Steinberger and Karel Jezek. [Using Latent Semantic Analysis in Text Summarization and Summary Evaluation](http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf). Proc. ISIM’04, 2004.
11. Josef Steinberger and Karel Ježek. [Text summarization and singular value decomposition](https://www.researchgate.net/profile/Karel_Jezek2/publication/226424326_Text_Summarization_and_Singular_Value_Decomposition/links/57233c1308ae586b21d87e66/Text-Summarization-and-Singular-Value-Decomposition.pdf). International Conference on Advances in Information Systems, 2004.
12. Josef Steinberger, Massimo Poesio, Mijail A Kabadjov and Karel Ježek. [Two uses of anaphora resolution in summarization](http://www.sensei-conversation.eu/wp-content/uploads/files/IPMpaper_official.pdf). Information Processing & Management, 2007.
13. James Clarke and Mirella Lapata. [Modelling Compression with Discourse Constraints](http://jamesclarke.net/media/papers/clarke-lapata-emnlp07.pdf). EMNLP-CoNLL, 2007.
14. Dan Gillick and Benoit Favre. [A Scalable Global Model for Summarization](https://pdfs.semanticscholar.org/a1a2/748e68d019815f1107fa19b0ab628b63928a.pdf). ACL, 2009.
15. Ani Nenkova and Kathleen McKeown. [Automatic summarization](https://www.cis.upenn.edu/~nenkova/1500000015-Nenkova.pdf).
Foundations and Trend in Information Retrieval, 2011. [The slides](https://www.fosteropenscience.eu/sites/default/files/pdf/2932.pdf) are also available.
16. Vahed Qazvinian, Dragomir R. Radev, Saif M. Mohammad, Bonnie Dorr, David Zajic, Michael Whidby, Taesun Moon. [Generating Extractive Summaries of Scientific Paradigms](https://arxiv.org/abs/1402.0556v1). arXiv:1402.0556, 2014.
17. Kågebäck, Mikael, et al. [Extractive summarization using continuous vector space models](http://www.aclweb.org/anthology/W14-1504). Proceedings of the 2nd Workshop on Continuous Vector Space Models and their Compositionality (CVSC)@ EACL. 2014.
18. Ramesh Nallapati, Bowen Zhou, Mingbo Ma. [Classify or Select: Neural Architectures for Extractive Document Summarization](https://arxiv.org/abs/1611.04244).  arXiv:1611.04244. 2016.
19. Ramesh Nallapati, Feifei Zhai, Bowen Zhou. [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230). arXiv:1611.04230, AAAI, 2017.
20. Shashi Narayan, Nikos Papasarantopoulos, Mirella Lapata, Shay B. Cohen. [Neural Extractive Summarization with Side Information](https://arxiv.org/abs/1704.04530). arXiv:1704.04530, 2017.
21. Rakesh Verma, Daniel Lee. [Extractive Summarization: Limits, Compression, Generalized Model and Heuristics](https://arxiv.org/abs/1704.05550v1). arXiv:1704.05550, 2017.
22. Ed Collins, Isabelle Augenstein, Sebastian Riedel. [A Supervised Approach to Extractive Summarisation of Scientific Papers](https://arxiv.org/abs/1706.03946v1). arXiv:1706.03946, 2017.
23. Sukriti Verma, Vagisha Nidhi. [Extractive Summarization using Deep Learning](https://arxiv.org/abs/1708.04439v1). arXiv:1708.04439, 2017.
24. Parth Mehta, Gaurav Arora, Prasenjit Majumder. [Attention based Sentence Extraction from Scientific Articles using Pseudo-Labeled data](https://arxiv.org/abs/1802.04675v1).     arXiv:1802.04675, 2018.
25. Shashi Narayan, Shay B. Cohen, Mirella Lapata. [Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/abs/1802.08636v1). arXiv:1802.08636, NAACL, 2018.
26. Aakash Sinha, Abhishek Yadav, Akshay Gahlot. [Extractive Text Summarization using Neural Networks](https://arxiv.org/abs/1802.10137v1). arXiv:1802.10137, 2018.
27. Yuxiang Wu, Baotian Hu. [Learning to Extract Coherent Summary via Deep Reinforcement Learning](https://arxiv.org/abs/1804.07036). arXiv:1804.07036, AAAI, 2018.

### Abstractive Text Summarization

1. Alexander M. Rush, Sumit Chopra, Jason Weston. [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685). EMNLP, 2015. The source code in LUA Torch7 is [NAMAS](https://github.com/facebook/NAMAS).
   - They use sequence-to-sequence encoder-decoder LSTM with attention.
   - They use the first sentence of a document. The source document is quite small (about 1 paragraph or ~500 words in the training dataset of Gigaword) and the produced output is also very short (about 75 characters). It remains an open challenge to scale up these limits - to produce longer summaries over multi-paragraph text input (even good LSTM models with attention models fall victim to vanishing gradients when the input sequences become longer than a few hundred items).
   - The evaluation method used for automatic summarization has traditionally been the ROUGE metric - which has been shown to correlate well with human judgment of summary quality, but also has a known tendency to encourage "extractive" summarization - so that using ROUGE as a target metric to optimize will lead a summarizer towards a copy-paste behavior of the input instead of the hoped-for reformulation type of summaries.
2. Peter Liu and Xin Pan. [Sequence-to-Sequence with Attention Model for Text Summarization](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html). 2016. The source code in Python is [textsum](https://github.com/tensorflow/models/tree/master/textsum).
   - They use sequence-to-sequence encoder-decoder LSTM with attention and bidirectional neural net.
   - They use the first 2 sentences of a document with a limit at 120 words.
   - The scores achieved by Google’s *textsum* are 42.57 ROUGE-1 and 23.13 ROUGE-2.
3. Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos santos, Caglar Gulcehre, Bing Xiang. [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023). arXiv:1602.06023, 2016.
   - They use GRU with attention and bidirectional neural net.
   - They use the first 2 sentences of a documnet with a limit at 120 words.
   - They use the [Large vocabulary trick (LVT)](https://arxiv.org/abs/1412.2007) of Jean et al. 2014, which means when you decode, use only the words that appear in the source - this reduces perplexity. But then you lose the capability to do "abstractive" summary. So they do "vocabulary expansion" by adding a layer of "word2vec nearest neighbors" to the words in the input.
   - Feature rich encoding - they add TFIDF and Named Entity types to the word embeddings (concatenated) to the encodings of the words - this adds to the encoding dimensions that reflect "importance" of the words. 
   - The most interesting of all is what they call the "Switching Generator/Pointer" layer. In the decoder, they add a layer that decides to either generate a new word based on the context / previously generated word (usual decoder) or copy a word from the input (that is - add a pointer to the input). They learn when to do Generate vs. Pointer and when it is a Pointer which word of the input to Point to.
4. Konstantin Lopyrev. [Generating News Headlines with Recurrent Neural Networks](https://arxiv.org/abs/1512.01712). arXiv:1512.01712, 2015. The source code in Python is [headlines](https://github.com/udibr/headlines).
5. Jiwei Li, Minh-Thang Luong and Dan Jurafsky. [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057). arXiv:1506.01057, 2015. The source code in Matlab is [Hierarchical-Neural-Autoencoder](https://github.com/jiweil/Hierarchical-Neural-Autoencoder).
6. Sumit Chopra, Alexander M. Rush and Michael Auli. [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://harvardnlp.github.io/papers/naacl16_summary.pdf). NAACL, 2016.
7. Jianpeng Cheng, Mirella Lapata. [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/abs/1603.07252). arXiv:1603.07252, 2016.
   - This paper uses attention as a mechanism for identifying the best sentences to extract, and then go beyond that to generate an abstractive summary.
8. Siddhartha Banerjee, Prasenjit Mitra, Kazunari Sugiyama. [Generating Abstractive Summaries from Meeting Transcripts](https://arxiv.org/abs/1609.07033v1). arXiv:1609.07033, Proceedings of the 2015 ACM Symposium on Document Engineering, DocEng' 2015.
9. Siddhartha Banerjee, Prasenjit Mitra, Kazunari Sugiyama. [Multi-document abstractive summarization using ILP based multi-sentence compression](https://arxiv.org/abs/1609.07034v1). arXiv:1609.07034, 2016.
10. Suzuki, Jun, and Masaaki Nagata. [Cutting-off Redundant Repeating Generations for Neural Abstractive Summarization](http://www.aclweb.org/anthology/E17-2047).  EACL 2017 (2017): 291.
11. Jiwei Tan and Xiaojun Wan. [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](). ACL, 2017.
12. Preksha Nema, Mitesh M. Khapra, Balaraman Ravindran and Anirban Laha. [Diversity driven attention model for query-based abstractive summarization](). ACL,2017
13. Romain Paulus, Caiming Xiong, Richard Socher. [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304). arXiv:1705.04304, 2017. The related blog is [Your tldr by an ai: a deep reinforced model for abstractive summarization](https://einstein.ai/research/your-tldr-by-an-ai-a-deep-reinforced-model-for-abstractive-summarization). 
    * Their model is trained with teacher forcing and reinforcement learning at the same time, being able to make use of both word-level and whole-summary-level supervision to make it more coherent and readable.
14. Shibhansh Dohare, Harish Karnick. [Text Summarization using Abstract Meaning Representation](https://arxiv.org/abs/1706.01678). arXiv:1706.01678, 2017.
15. Piji Li, Wai Lam, Lidong Bing, Zihao Wang. [Deep Recurrent Generative Decoder for Abstractive Text Summarization](https://arxiv.org/abs/1708.00625v1). arXiv:1708.00625, 2017.
16. Xinyu Hua, Lu Wang. [A Pilot Study of Domain Adaptation Effect for Neural Abstractive Summarization](https://arxiv.org/abs/1707.07062v1). arXiv:1707.07062, 2017.
17. Angela Fan, David Grangier, Michael Auli. [Controllable Abstractive Summarization](https://arxiv.org/abs/1711.05217v1). arXiv:1711.05217, 2017.
18. Linqing Liu, Yao Lu, Min Yang, Qiang Qu, Jia Zhu, Hongyan Li. [Generative Adversarial Network for Abstractive Text Summarization](https://arxiv.org/abs/1711.09357v1). arXiv:1711.09357, 2017.
19. Johan Hasselqvist, Niklas Helmertz, Mikael Kågebäck. [Query-Based Abstractive Summarization Using Neural Networks](https://arxiv.org/abs/1712.06100v1). arXiv:1712.06100, 2017.
20. Tal Baumel, Matan Eyal, Michael Elhadad. [Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704v2). arXiv:1801.07704, 2018.
21. André Cibils, Claudiu Musat, Andreea Hossman, Michael Baeriswyl. [Diverse Beam Search for Increased Novelty in Abstractive Summarization](https://arxiv.org/abs/1802.01457v1). arXiv:1802.01457, 2018.
22. Chieh-Teng Chang, Chi-Chia Huang, Jane Yung-Jen Hsu. [A Hybrid Word-Character Model for Abstractive Summarization](https://arxiv.org/abs/1802.09968v1). arXiv:1802.09968, 2018.
23. Asli Celikyilmaz, Antoine Bosselut, Xiaodong He, Yejin Choi. [Deep Communicating Agents for Abstractive Summarization](https://arxiv.org/abs/1803.10357v1). arXiv:1803.10357, 2018.
24. Piji Li, Lidong Bing, Wai Lam. [Actor-Critic based Training Framework for Abstractive Summarization](https://arxiv.org/abs/1803.11070v1). arXiv:1803.11070, 2018.
25. Paul Azunre, Craig Corcoran, David Sullivan, Garrett Honke, Rebecca Ruppel, Sandeep Verma, Jonathon Morgan. [Abstractive Tabular Dataset Summarization via Knowledge Base Semantic Embeddings](https://arxiv.org/abs/1804.01503v2). arXiv:1804.01503, 2018.
26. Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, Nazli Goharian. [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685v1). arXiv:1804.05685, 2018.
27. Ramakanth Pasunuru, Mohit Bansal. [Multi-Reward Reinforced Summarization with Saliency and Entailment](https://arxiv.org/abs/1804.06451v1). arXiv:1804.06451, 2018.
28. Shuming Ma, Xu Sun, Junyang Lin, Xuancheng Ren. [A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification](https://arxiv.org/abs/1805.01089). arXiv:1805.01089, IJCAI 2018.
28. Li Wang, Junlin Yao, Yunzhe Tao, Li Zhong, Wei Liu, Qiang Du. [A Reinforced Topic-Aware Convolutional Sequence-to-Sequence Model for Abstractive Text Summarization](https://arxiv.org/abs/1805.03616). arXiv:1805.03616, International Joint Conference on Artificial Intelligence and European Conference on Artificial Intelligence (IJCAI-ECAI), 2018.

### Text Summarization

1. Eduard Hovy and Chin-Yew Lin. [Automated text summarization and the summarist system](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/tipster-proc-hovy-lin-final.pdf). In Proceedings of a Workshop on Held at Baltimore, Maryland, ACL, 1998.
2. Eduard Hovy and Chin-Yew Lin. [Automated Text Summarization in SUMMARIST](https://www.isi.edu/natural-language/people/hovy/papers/98hovylin-summarist.pdf). In Advances in Automatic Text Summarization, 1999.
3. Dipanjan Das and Andre F.T. Martins. [A survey on automatic text summarization](https://wtlab.um.ac.ir/images/e-library/text_summarization/A%20Survey%20on%20Automatic%20Text%20Summarization.pdf). Technical report, CMU, 2007
4. J. Leskovec, L. Backstrom, J. Kleinberg. [Meme-tracking and the Dynamics of the News Cycle](http://www.memetracker.org). ACM SIGKDD Intl. Conf. on Knowledge Discovery and Data Mining, 2009.
5. Ryang, Seonggi, and Takeshi Abekawa. "[Framework of automatic text summarization using reinforcement learning](http://dl.acm.org/citation.cfm?id=2390980)." In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 256-265. Association for Computational Linguistics, 2012. [not neural-based methods]
6. King, Ben, Rahul Jha, Tyler Johnson, Vaishnavi Sundararajan, and Clayton Scott. "[Experiments in Automatic Text Summarization Using Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8775&rep=rep1&type=pdf)." Machine Learning (2011).
7. Liu, Yan, Sheng-hua Zhong, and Wenjie Li. "[Query-Oriented Multi-Document Summarization via Unsupervised Deep Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5058/5322)." AAAI. 2012.
8. He, Zhanying, Chun Chen, Jiajun Bu, Can Wang, Lijun Zhang, Deng Cai, and Xiaofei He. "[Document Summarization Based on Data Reconstruction](http://cs.nju.edu.cn/zlj/pdf/AAAI-2012-He.pdf)." In AAAI. 2012.
9. Mohsen Pourvali, Mohammad Saniee Abadeh. [Automated Text Summarization Base on Lexicales Chain and graph Using of WordNet and Wikipedia Knowledge Base](https://arxiv.org/abs/1203.3586v1). arXiv:1203.3586, 2012.
10. PadmaPriya, G., and K. Duraiswamy. [An Approach For Text Summarization Using Deep Learning Algorithm](http://thescipub.com/PDF/jcssp.2014.1.9.pdf). Journal of Computer Science 10, no. 1 (2013): 1-9.
11. Rushdi Shams, M.M.A. Hashem, Afrina Hossain, Suraiya Rumana Akter, Monika Gope. [Corpus-based Web Document Summarization using Statistical and Linguistic Approach](https://arxiv.org/abs/1304.2476v1). arXiv:1304.2476, Procs. of the IEEE International Conference on Computer and Communication Engineering (ICCCE10), pp. 115-120, Kuala Lumpur, Malaysia, May 11-13, (2010).
12. Juan-Manuel Torres-Moreno. [Beyond Stemming and Lemmatization: Ultra-stemming to Improve Automatic Text Summarization](https://arxiv.org/abs/1209.3126). arXiv:1209.3126, 2012.
13. Rioux, Cody, Sadid A. Hasan, and Yllias Chali. [Fear the REAPER: A System for Automatic Multi-Document Summarization with Reinforcement Learning](http://emnlp2014.org/papers/pdf/EMNLP2014075.pdf). In EMNLP, pp. 681-690. 2014.[not neural-based methods]
14. Fatma El-Ghannam, Tarek El-Shishtawy. [Multi-Topic Multi-Document Summarizer](https://arxiv.org/abs/1401.0640v1). arXiv:1401.0640, 2014.
15. Denil, Misha, Alban Demiraj, and Nando de Freitas. [Extraction of Salient Sentences from Labelled Documents](http://arxiv.org/abs/1412.6815). arXiv:1412.6815,  2014.
16. Denil, Misha, Alban Demiraj, Nal Kalchbrenner, Phil Blunsom, and Nando de Freitas.[Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network](http://arxiv.org/abs/1406.3830). arXiv:1406.3830, 2014.
17. Cao, Ziqiang, Furu Wei, Li Dong, Sujian Li, and Ming Zhou. [Ranking with Recursive Neural Networks and Its Application to Multi-document Summarization](http://gana.nlsde.buaa.edu.cn/~lidong/aaai15-rec_sentence_ranking.pdf). AAAI, 2015.
18. Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, and Noah A. Smith. [Toward Abstractive Summarization Using Semantic Representations](http://www.cs.cmu.edu/~nasmith/papers/liu+flanigan+thomson+sadeh+smith.naacl15.pdf). NAACL, 2015.
19. Wenpeng Yin， Yulong Pei. Optimizing Sentence Modeling and Selection for Document Summarization. IJCAI, 2015.
20. Liu, He, Hongliang Yu, and Zhi-Hong Deng. [Multi-Document Summarization Based on Two-Level Sparse Representation Model](http://www.cis.pku.edu.cn/faculty/system/dengzhihong/papers/AAAI%202015_Multi-Document%20Summarization%20Based%20on%20Two-Level%20Sparse%20Representation%20Model.pdf). In Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
21. Jin-ge Yao, Xiaojun Wan and Jianguo Xiao. [Compressive Document Summarization via Sparse Optimization](http://ijcai.org/Proceedings/15/Papers/198.pdf). IJCAI, 2015.
22. Piji Li, Lidong Bing, Wai Lam, Hang Li, and Yi Liao. [Reader-Aware Multi-Document Summarization via Sparse Coding](http://arxiv.org/abs/1504.07324). arXiv:1504.07324,  IJCAI, 2015.
23. Marta Aparício, Paulo Figueiredo, Francisco Raposo, David Martins de Matos, Ricardo Ribeiro, Luís Marujo. [Summarization of Films and Documentaries Based on Subtitles and Scripts](https://arxiv.org/abs/1506.01273v3). arXiv:1506.01273, 2015.
24. Luís Marujo, Ricardo Ribeiro, David Martins de Matos, João P. Neto, Anatole Gershman, Jaime Carbonell. [Extending a Single-Document Summarizer to Multi-Document: a Hierarchical Approach](https://arxiv.org/abs/1507.02907v1). arXiv:1507.02907, 2015.
25. Xiaojun Wan, Yansong Feng and Weiwei Sun. [Automatic Text Generation: Research Progress and Future Trends](http://www.icst.pku.edu.cn/lcwm/wanxj/files/TextGenerationSurvey.pdf). Book Chapter in CCF 2014-2015 Annual Report on Computer Science and Technology in China (In Chinese), 2015.
    0. Xiaojun Wan, Ziqiang Cao, Furu Wei, Sujian Li, Ming Zhou. [Multi-Document Summarization via Discriminative Summary Reranking](https://arxiv.org/abs/1507.02062v1).     arXiv:1507.02062, 2015.
26. Gulcehre, Caglar, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. [Pointing the Unknown Words](http://arxiv.org/abs/1603.08148). arXiv:1603.08148, 2016.
27. Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393). arXiv:1603.06393, ACL, 2016.
    - They addressed an important problem in sequence-to-sequence (Seq2Seq) learning referred to as copying, in which certain segments in the input sequence are selectively replicated in the output sequence. In this paper, they incorporated copying into neural network-based Seq2Seq learning and propose a new model called CopyNet with encoder-decoder structure. CopyNet can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose sub-sequences in the input sequence and put them at proper places in the output sequence.
28. Jianmin Zhang, Jin-ge Yao and Xiaojun Wan. [Toward constructing sports news from live text commentary](http://www.icst.pku.edu.cn/lcwm/wanxj/files/acl16_sports.pdf). In Proceedings of ACL, 2016.
29. Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. "[AttSum: Joint Learning of Focusing and Summarization with Neural Attention](http://arxiv.org/abs/1604.00125)".  arXiv:1604.00125, 2016
30. Ayana, Shiqi Shen, Yu Zhao, Zhiyuan Liu and Maosong Sun. [Neural Headline Generation with Sentence-wise Optimization](https://arxiv.org/abs/1604.01904). arXiv:1604.01904, 2016.
31. Ayana, Shiqi Shen, Zhiyuan Liu and Maosong Sun. [Neural Headline Generation with Minimum Risk Training](https://128.84.21.199/abs/1604.01904v1). 2016.
32. Lu Wang, Hema Raghavan, Vittorio Castelli, Radu Florian, Claire Cardie. [A Sentence Compression Based Framework to Query-Focused Multi-Document Summarization](https://arxiv.org/abs/1606.07548v1). arXiv:1606.07548, 2016.
33. Milad Moradi, Nasser Ghadiri. [Different approaches for identifying important concepts in probabilistic biomedical text summarization](https://arxiv.org/abs/1605.02948v3). arXiv:1605.02948, 2016.
34. Kikuchi, Yuta, Graham Neubig, Ryohei Sasano, Hiroya Takamura, and Manabu Okumura. [Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552). arXiv:1609.09552, 2016.
8. Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei and Hui Jiang. [Distraction-Based Neural Networks for Document Summarization](https://arxiv.org/abs/1610.08462). arXiv:1610.08462, IJCAI, 2016.
35. Wang, Lu, and Wang Ling. [Neural Network-Based Abstract Generation for Opinions and Arguments](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf). NAACL, 2016.
36. Yishu Miao, Phil Blunsom.  [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](http://arxiv.org/abs/1609.07317).  EMNLP, 2016.
37. Takase, Sho, Jun Suzuki, Naoaki Okazaki, Tsutomu Hirao, and Masaaki Nagata. [Neural headline generation on abstract meaning representation](https://www.aclweb.org/anthology/D/D16/D16-1112.pdf).  EMNLP, 1054-1059, 2016.
38. Wenyuan Zeng, Wenjie Luo, Sanja Fidler, Raquel Urtasun.  [Efficient Summarization with Read-Again and Copy Mechanism](https://arxiv.org/abs/1611.03382). arXiv:1611.03382, 2016.
39. Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. [Improving Multi-Document Summarization via Text Classification](https://arxiv.org/abs/1611.09238v1). arXiv:1611.09238, 2016.
40. Hongya Song, Zhaochun Ren, Piji Li, Shangsong Liang, Jun Ma, and Maarten de Rijke. [Summarizing Answers in Non-Factoid Community Question-Answering](http://dl.acm.org/citation.cfm?id=3018704). In WSDM 2017: The 10th International Conference on Web Search and Data Mining, 2017.
41. Piji Li, Zihao Wang, Wai Lam, Zhaochun Ren, Lidong Bing.  [Salience Estimation via Variational Auto-Encoders for Multi-Document Summarization](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14613). In AAAI, 2017.
42. Yinfei Yang, Forrest Sheng Bao, Ani Nenkova. [Detecting (Un)Important Content for Single-Document News Summarization](https://arxiv.org/abs/1702.07998v1). arXiv:1702.07998, 2017.
43. Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky, Yu Chi. [Deep Keyphrase Generation](https://arxiv.org/abs/1704.06879v1). arXiv:1704.06879, 2017. The source code written in Python is [seq2seq-keyphrase](https://github.com/memray/seq2seq-keyphrase).
44. Abigail See, Peter J. Liu and Christopher D. Manning. [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368). ACL, 2017.
45. Qingyu Zhou, Nan Yang, Furu Wei and Ming Zhou. [Selective Encoding for Abstractive Sentence Summarization](https://arxiv.org/abs/1704.07073). arXiv:1704.07073, ACL, 2017.
46. Maxime Peyrard and Judith Eckle-Kohler. [Supervised Learning of Automatic Pyramid for Optimization-Based Multi-Document Summarization](). ACL, 2017.
47. Jin-ge Yao, Xiaojun Wan and Jianguo Xiao. [Recent Advances in Document Summarization](http://www.icst.pku.edu.cn/lcwm/wanxj/files/summ_survey_draft.pdf). KAIS, survey paper, 2017.
48. Pranay Mathur, Aman Gill and Aayush Yadav. [Text Summarization in Python: Extractive vs. Abstractive techniques revisited](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/#text_summarization_in_python). 2017.
    - They compared modern extractive methods like LexRank, LSA, Luhn and Gensim’s existing TextRank summarization module on the [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) of 51 (article, summary) pairs. They also had a try with an abstractive technique using Tensorflow’s algorithm [textsum](https://github.com/tensorflow/models/tree/master/textsum), but didn’t obtain good results due to its extremely high hardware demands (7000 GPU hours).
49. Arman Cohan, Nazli Goharian. [Scientific Article Summarization Using Citation-Context and Article's Discourse Structure](https://arxiv.org/abs/1704.06619v1). arXiv:1704.06619, EMNLP, 2015.
50. Arman Cohan, Nazli Goharian. [Scientific document summarization via citation contextualization and scientific discourse](https://arxiv.org/abs/1706.03449v1). arXiv:1706.03449, 2017.
51. Michihiro Yasunaga, Rui Zhang, Kshitijh Meelu, Ayush Pareek, Krishnan Srinivasan, Dragomir Radev. [Graph-based Neural Multi-Document Summarization](https://arxiv.org/abs/1706.06681v3). arXiv:1706.06681, CoNLL, 2017.
52. Abeed Sarker, Diego Molla, Cecile Paris. [Automated text summarisation and evidence-based medicine: A survey of two domains](https://arxiv.org/abs/1706.08162v1). arXiv:1706.08162, 2017.
53. Mehdi Allahyari, Seyedamin Pouriyeh, Mehdi Assefi, Saeid Safaei, Elizabeth D. Trippe, Juan B. Gutierrez, Krys Kochut. [Text Summarization Techniques: A Brief Survey](https://arxiv.org/abs/1707.02268). arXiv:1707.02268, 2017.
    5. Demian Gholipour Ghalandari. [Revisiting the Centroid-based Method: A Strong Baseline for Multi-Document Summarization](https://arxiv.org/abs/1708.07690v1).     arXiv:1708.07690, EMNLP, 2017.
54. Shuming Ma, Xu Sun. [A Semantic Relevance Based Neural Network for Text Summarization and Text Simplification](https://arxiv.org/abs/1710.02318v1). arXiv:1710.02318, 2017.
55. Kaustubh Mani, Ishan Verma, Lipika Dey. [Multi-Document Summarization using Distributed Bag-of-Words Model](https://arxiv.org/abs/1710.02745v1). arXiv:1710.02745, 2017.
56. Liqun Shao, Hao Zhang, Ming Jia, Jie Wang. [Efficient and Effective Single-Document Summarizations and A Word-Embedding Measurement of Quality](https://arxiv.org/abs/1710.00284v1). arXiv:1710.00284, KDIR, 2017.
57. Mohammad Ebrahim Khademi, Mohammad Fakhredanesh, Seyed Mojtaba Hoseini. [Conceptual Text Summarizer: A new model in continuous vector space](https://arxiv.org/abs/1710.10994v2). arXiv:1710.10994, 2017.
58. Jingjing Xu. [Improving Social Media Text Summarization by Learning Sentence Weight Distribution](https://arxiv.org/abs/1710.11332v1). arXiv:1710.11332, 2017.
59. Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, Noam Shazeer. [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198v1). arXiv:1801.10198, 2018.
60. Parth Mehta, Prasenjit Majumder. [Content based Weighted Consensus Summarization](https://arxiv.org/abs/1802.00946v1). arXiv:1802.00946, 2018.
61. Mayank Chaudhari, Aakash Nelson Mattukoyya. [Tone Biased MMR Text Summarization](https://arxiv.org/abs/1802.09426v2). arXiv:1802.09426, 2018.

### Chinese Text Summarization

1. Mao Song Sun. [Natural Language Processing Based on Naturally Annotated Web Resources](http://www.thunlp.org/site2/images/stories/files/2011_zhongwenxinxixuebao_sms.pdf). Journal of Chinese Information Processing, 2011.
2. Baotian Hu, Qingcai Chen and Fangze Zhu. [LCSTS: A Large Scale Chinese Short Text Summarization Dataset](https://arxiv.org/abs/1506.05865). 2015.
   - They constructed a large-scale Chinese short text summarization dataset constructed from the Chinese microblogging website Sina Weibo, which is released to [the public](http://icrc.hitsz.edu.cn/Article/show/139.html). Then they performed GRU-based encoder-decoder method on it to generate summary. They took the whole short text as one sequence, this may not be very reasonable, because most of short texts contain several sentences.
   - LCSTS contains 2,400,591 (short text, summary) pairs as the training set and 1,106  pairs as the test set.
   - All the models are trained on the GPUs tesla M2090 for about one week.
   - The results show that the RNN with context outperforms RNN without context on both character and word based input.
   - Moreover, the performances of the character-based input outperform the word-based input.

### Evaluation Metrics

1. Chin-Yew Lin and Eduard Hovy. [Automatic Evaluation of Summaries Using N-gram
Co-Occurrence Statistics](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/naacl2003.pdf). In Proceedings of the Human Technology Conference 2003 (HLT-NAACL-2003).
2. Chin-Yew Lin. [Rouge: A package for automatic evaluation of summaries](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/was2004.pdf). Workshop on Text Summarization Branches Out, Post-Conference Workshop of ACL 2004.
3. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
4. Arman Cohan, Nazli Goharian. [Revisiting Summarization Evaluation for Scientific Articles](https://arxiv.org/abs/1604.00400v1). arXiv:1604.00400, LREC, 2016.
5. Maxime Peyrard. [A Formal Definition of Importance for Summarization](https://arxiv.org/abs/1801.08991v1). arXiv:1801.08991, 2018.
6. Kavita Ganesan. [ROUGE 2.0: Updated and Improved Measures for Evaluation of Summarization Tasks](https://arxiv.org/abs/1803.01937v1). arXiv:1803.01937, 2018. It works by comparing an automatically produced summary or translation against a set of reference summaries (typically human-produced).  ROUGE is one of the standard ways to compute effectiveness of auto generated summaries. The evaluation toolkit [ROUGE 2.0](https://github.com/RxNLP/ROUGE-2.0) is an easy to use  for Automatic Summarization tasks. 

### Opinion Summarization

1. Kavita Ganesan, ChengXiang Zhai and Jiawei Han. [Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions](http://kavita-ganesan.com/opinosis). Proceedings of COLING '10, 2010.
2. Kavita Ganesan, ChengXiang Zhai and Evelyne Viegas. [Micropinion Generation: An Unsupervised Approach to Generating Ultra-Concise Summaries of Opinions](http://kavita-ganesan.com/micropinion-generation). WWW'12, 2012.
3. Kavita Ganesan. [Opinion Driven Decision Support System (ODSS)](http://kavita-ganesan.com/phd-thesis). PhD Thesis, University of Illinois at Urbana-Champaign, 2013.
4. Ozan Irsoy and Claire Cardie. [Opinion Mining with Deep Recurrent Neural Networks](https://www.cs.cornell.edu/~oirsoy/files/emnlp14drnt.pdf). In EMNLP, 2014.
5. Ahmad Kamal. [Review Mining for Feature Based Opinion Summarization and Visualization](https://arxiv.org/abs/1504.03068v2). arXiv:1504.03068, 2015.
6. Haibing Wu, Yiwei Gu, Shangdi Sun and Xiaodong Gu. [Aspect-based Opinion Summarization with Convolutional Neural Networks](https://arxiv.org/abs/1511.09128). 2015.
7. Lu Wang, Hema Raghavan, Claire Cardie, Vittorio Castelli. [Query-Focused Opinion Summarization for User-Generated Content](https://arxiv.org/abs/1606.05702v1). arXiv:1606.05702, 2016.
### Support:

If you want the good work to continue please support us on

* [PAYPAL](https://www.paypal.me/ishandutta2007)
* [BITCOIN ADDRESS: 3LZazKXG18Hxa3LLNAeKYZNtLzCxpv1LyD](https://www.coinbase.com/join/5a8e4a045b02c403bc3a9c0c)
