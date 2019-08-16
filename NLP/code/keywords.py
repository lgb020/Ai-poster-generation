import util as utils
from heapq import nlargest
from itertools import count
import numpy as  np

class Keywords(object):
    def __init__(self,
                 use_stopword=True,
                 stop_words_file=utils.default_stopwords_file(),
                 max_iter=100,
                 tol=0.0001,
                 window=2):
        self.__use_stopword = use_stopword
        self.__max_iter = max_iter
        self.__tol = tol
        self.__window = window
        self.__stop_words = set()
        self.__stop_words_file = utils.default_stopwords_file()
        if stop_words_file:
            self.__stop_words_file = stop_words_file
        if use_stopword:
            with open(self.__stop_words_file, 'r', encoding='utf-8') as f:
                for word in f:
                    self.__stop_words.add(word.strip())
        np.seterr(all='warn')

    @staticmethod
    def build_vocab(sents):
        word_index = {}
        index_word = {}
        words_number = 0
        for word_list in sents:
            for word in word_list:
                if word not in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1
        return word_index, index_word, words_number

    @staticmethod
    def create_graph(sents, words_number, word_index, window=2):
        graph = [[0.0 for _ in range(words_number)] for _ in range(words_number)]
        for word_list in sents:
            for w1, w2 in utils.combine(word_list, window):
                if w1 in word_index and w2 in word_index:
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    graph[index1][index2] += 1.0
                    graph[index2][index1] += 1.0
        return graph

    def keywords(self, text, n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = utils.as_text(text)
        tokens = utils.cut_sentences(text)
        sentences, sents = utils.psegcut_filter_words(tokens,
                                                      self.__stop_words,
                                                      self.__use_stopword)

        word_index, index_word, words_number = self.build_vocab(sents)
        graph = self.create_graph(sents, words_number,
                                  word_index, window=self.__window)
        scores = utils.weight_map_rank(graph, max_iter=self.__max_iter,
                                       tol=self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            sent_index.append(sent_selected[i][1])
        return [index_word[i] for i in sent_index]

