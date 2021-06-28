#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmoputh.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import codecs
import json
import os
import time
from tqdm import tqdm
import pandas as pd
import colorama
from collections import Counter
from multiprocessing import Pool, cpu_count

import nltk
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.api import CorpusReader

DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
CAT_PATTERN = r'([a-z_\s]+)/.*'


class MediaCloud_DataReader(CategorizedCorpusReader, CorpusReader):
    def __init__(self, corpus_root, fileids=DOC_PATTERN, encoding='utf-8', **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, corpus_root, fileids, encoding)

    def resolve(self, fileids=None, categories=None):
        """
        return file ids given explicit fileids or categories
        :param fileids:
        :param categories:
        :return:
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)  # return a list of identifiers (json path) in this corpus
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an HTML document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        # abspaths will return a list of all file identifiers in this corpus
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield json.load(f)  # will generate new dicts

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def get_title(self, fileids=None, categories=None):
        """
        Return the media name of each document, for further lable the document
        """
        for doc in self.docs(fileids, categories):
            yield doc['title']

    def get_media(self, fileids=None, categories=None):
        """
        Return the media name of each document, for further lable the document
        """
        for doc in self.docs(fileids, categories):
            yield doc['media']

    def get_author(self, fileids=None, categories=None):
        """
        Return the media name of each document, for further lable the document
        """
        for doc in self.docs(fileids, categories):
            yield doc['author']

    def get_media_label(self, fileids=None, categories=None):
        """
        Return the media name of each document, for further lable the document
        """
        for doc in self.docs(fileids, categories):
            me = doc['media']
            if me in ['BBC', 'CNN', 'New York Times', 'NPR', 'Washington Post', 'HuffPost', 'guardiannews.com']:
                # Liberal
                yield 0
            elif me in ['CNBC', 'USA Today', 'Wall Street Journal', 'CBS News', 'ABC.com']:
                # Neutral
                yield 1
            elif me in ['rushlimbaugh.com', 'The Sean Hannity Show', 'Fox News', 'Breitbart']:
                # Conservative
                yield 2
            else:
                print(me)
                yield 2

            # yield doc['media']

    def get_pubdate(self, fileids=None, categories=None):
        """
        Return the media name of each document, for further lable the document
        """
        for doc in self.docs(fileids, categories):
            yield doc['pub_date']

    def get_keywords(self, fileids=None, categories=None):
        """
        Return the media name of each document, for further lable the document
        """
        for doc in self.docs(fileids, categories):
            yield doc['keywords']

    def clean_text(self, fileids=None, categories=None):
        """
        Returns the HTML content of each document, cleaning it using
        the readability-lxml library.
        """
        for doc in self.docs(fileids, categories):
            yield doc['text']

    def paras(self, fileids=None, categories=None):
        """
        Uses BeautifulSoup to parse the paragraphs from the HTML.
        """
        for text in self.clean_text(fileids, categories):
            for para in text.split('\n\n'):
                yield para

    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def __len__(self, fileids=None, categories=None):
        return len(self.resolve(fileids, categories) or self.fileids())

    def show_stats(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time.time()

        # Structures to perform counting.
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize_pos and count
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in sent_tokenize(para):
                counts['sents'] += 1

                for word in wordpunct_tokenize(sent):
                    counts['words'] += 1
                    tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        print(
            "files: {}, topics: {}, paras: {}, sents: {}, words: {}, vocab: {},"
            " lexdiv: {:0.2f}, ppdoc: {:0.2f}, sppar: {:0.2f}, secs: {:0.2f}".format(
                n_fileids, n_topics, counts['paras'],
                counts['sents'], counts['words'],
                len(tokens), float(counts['words']) / float(len(tokens)), float(counts['paras']) / float(n_fileids),
                float(counts['sents']) / float(counts['paras']), time.time() - started))


class Preprocessor(object):
    def __init__(self, corpus, target=None, **kwargs):
        """
        convert the corpus to dataframe, with corresponding attributes filled
        :param corpus:
        :param out_name:
        :param target:
        :param kwargs:
        """
        self.corpus = corpus
        self.target = target

    def get_fileids_size(self, fileids=None, categories=None):
        return len(self.corpus.resolve(fileids, categories))

    def get_fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def tokenize_pos(self, fileid):
        """
        returns a generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def plain_text(self, fileid):
        """
        return list of paragraphs for each article
        :param fileid:
        :return:
        """
        yield [para for para in self.corpus.paras(fileids=fileid)]

    def process(self, fileid):
        """
        single file processing function (given file id)
        :param fileid:
        :return: a dict with attributes filled in
        """

        # Create a data structure for the pickle
        document = {}
        document.update({'title': list(self.corpus.get_title(fileid))})
        document.update({'author': list(self.corpus.get_author(fileid))})
        document.update({'media': list(self.corpus.get_media(fileid))})
        document.update({'media_label': list(self.corpus.get_media_label(fileid))})
        document.update({'pubdate': list(self.corpus.get_pubdate(fileid))})
        document.update({'words_pos': list(self.tokenize_pos(fileid))})
        document.update({'words': list(self.plain_text(fileid))})

        return document

    def transform(self, fileids=None, categories=None, thread_num=2):
        """
        multi-thread transforming
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        with Pool(thread_num) as proc:
            for cate in categories:
                print("preprocessing topic:", cate)
                results = list(
                    tqdm(proc.imap(self.process, self.get_fileids(fileids, cate), ),
                         total=self.get_fileids_size(categories=cate)))

                _df = pd.DataFrame(results)
                _df.to_csv(os.path.join(self.target, str(cate).lower().replace(' ', '_') + '.csv'), index=False)
                del _df
                print()


if __name__ == '__main__':
    colorama.init(autoreset=True)

    # picked_category = ['drones', 'abortion']  # if you want all topics, set to None
    picked_category = None
    processes_num = 20
    processes = min(processes_num, cpu_count())

    print()
    for year in ['2020']:
        print(colorama.Fore.LIGHTBLUE_EX + "= Loading the dataset ... " + "Year: " + year)
        corpus = MediaCloud_DataReader('./output_' + year)

        category = picked_category if picked_category else corpus.categories()
        print()
        print(colorama.Fore.LIGHTBLUE_EX + "= Showing the dataset statistics:")
        # corpus.show_stats(categories=category)
        print()

        # # very slow, unless you are really curious
        # words = Counter(corpus.words())
        # print("{:,} vocabulary {:,} word count".format(len(words.keys()), sum(words.values())))

        # save to a csv

        since = time.time()
        preprocessor = Preprocessor(corpus, './csv_output_' + year + '/')
        print(colorama.Fore.LIGHTBLUE_EX + "= Preprocessing ...")
        preprocessor.transform(thread_num=processes, categories=category)
        print()

        time_elapsed = time.time() - since
        print(colorama.Fore.LIGHTGREEN_EX + "= Preprocessing is done!")
        print('It takes {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
