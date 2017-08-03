from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__author__ = 'ivishnevskiy'

import re
import string
import sys



class Tokenizer():

    def __init__(self):
        if sys.version_info < (3,):
            self.maketrans = string.maketrans
        else:
            self.maketrans = str.maketrans



    def words_into_pairs(self, sentence):
        _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

        split = " "
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        text = sentence.translate(self.maketrans(filters, split * len(filters)))
        print(text)

        words = []
        for space_separated_fragment in text.strip().split():
            words.extend(_WORD_SPLIT.split(space_separated_fragment))


        print(words)
        #print [w for w in words if w]

        return [(w,words[idx+1]) for idx, w in enumerate(words) if idx+1 < len(words)]



    def words_into_pairs_no_cleaning(self, sentence):
        _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(_WORD_SPLIT.split(space_separated_fragment))


        print(words)
        return [w for w in words if w]


Tokenizer()
