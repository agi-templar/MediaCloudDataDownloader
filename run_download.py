#! /usr/bin/env python3
# coding=utf-8

# Author: Ruibo Liu (ruibo.liu.gr@dartmouth.edu)
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


import csv
import datetime
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace
from collections import namedtuple, Counter
from concurrent import futures
import mediacloud.tags
import mediacloud.api
import newspaper
import requests
from ftfy import fix_text
from newspaper import Article
from tqdm import tqdm
from media_list import Media

import socks
import requests
import socket

from utils import handle_dirs

MAX_WORKERS = 1000
MAX_CPUS = multiprocessing.cpu_count()

Story = namedtuple('Story', ['id',
                             'title',
                             'author',
                             'media',
                             'media_url',
                             'pub_date',
                             'stories_id',
                             'guid',
                             'processed_stories_id',
                             'text'])

Response = namedtuple('Response', ['response', 'status'])
Responses = namedtuple('Responses', ['responses', 'count'])


args = Namespace(
    load_dir='./',
    save_dir='./output_test/',
    output_dir='./output_new/',
    opml_file_name='feedly_new.opml',
    rss_format='xml',
    output_format='html',
    json_file_name='opml.json',
    expand_filepaths_to_save_dir=True,
)


def get_list_of_APIs(path):
    apis = []
    with open(path, 'r') as f:
        for line in f:
            apis.append(list(line.strip('\n').split(',')))
    return apis


def save_as_csv(save_dir, csv_file_name, content):
    """
    Save the content to a csv file
    :param save_dir: the saving directory
    :param csv_file_name: hashed id
    :param content: (namedtuple Story)
    :return:
    """
    handle_dirs(save_dir)
    csv_file_path = os.path.join(save_dir, csv_file_name)
    with open(csv_file_path, 'w+') as fp:
        f_csv = csv.DictWriter(fp, content._fields)
        f_csv.writeheader()
        f_csv.writerow(content._asdict())


def save_as_json(save_dir, json_file_name, content):
    """
    Save the content to a json file
    :param save_dir: the saving directory
    :param json_file_name: the json file name
    :param content: the content to be saved
    :return:
    """
    handle_dirs(save_dir)
    json_file_path = os.path.join(save_dir, json_file_name)
    fp = open(json_file_path, 'w+')
    fp.write(json.dumps(content._asdict(), indent=2))
    fp.close()
    # print("Save as json successfully!")


def save_as_txt(save_dir, txt_file_name, content):
    """
    Save the content to a json file
    :param save_dir: the saving directory
    :param txt_file_name: the txt file name
    :param content: the content to be saved
    :return:
    """
    handle_dirs(save_dir)
    txt_file_path = os.path.join(save_dir, txt_file_name)
    fp = open(txt_file_path, 'w+')
    fp.write(json.dumps(content, indent=2))
    fp.close()
    # print("Save as txt successfully!")


def set_themes(stories):
    """
    set the theme attr for each story
    :param stories:
    :return:
    """
    for s in stories:
        theme_tag_names = ','.join(
            [t['tag'] for t in s['story_tags'] if t['tag_sets_id'] == mediacloud.tags.TAG_SET_NYT_THEMES])
        s['themes'] = theme_tag_names
    return stories


def stories_about_topic(api_gen, mc, query, period, fetch_size=10, limit=10):
    """
    Return stories on certain topic from certain source, from start_time to end_time.
    :param mc: the media cloud client
    :param query: the query string
    :param period: the requested time period
    :param fetch_size:
    :param limit: max number of return stories
    :return: a list of stories
    """

    more_stories = True
    stories = []
    last_id = 0
    fetched_stories = []

    while more_stories:
        try:
            fetched_stories = mc.storyList(query, period, last_id, rows=fetch_size, sort='processed_stories_id')
        except mediacloud.error.MCException as e:
            if e.status_code == 429:
                print()
                print("Switch media cloud account!")
                print()
                mc = mediacloud.api.MediaCloud(next(api_gen)[0])  # call the generator
        if len(fetched_stories) == 0 or len(stories) > limit:
            more_stories = False
        else:
            stories += fetched_stories
            last_id = fetched_stories[-1]['processed_stories_id']
    stories = set_themes(stories)
    return stories


def get_one_article(story, cur_topic, save_format='json'):
    """
    Return a dict that stores all the information extracted from url
    :param cur_topic: current query topic
    :param save_format: 'json' or 'txt', as file format
    :param story: (story) a object from media cloud
    :return: the text of the story
    """
    response = Response
    article = Article(story['url'])

    if not article.is_media_news():
        try:
            article.download()
            article.parse()
        except newspaper.ArticleException:
            status = "fail"
            return Response(response, status)
        else:
            text = fix_text(article.text)

            # if no exception, set status to success
            status = 'success'

            # set attributes that story already has
            title = story['title']
            media_name = story['media_name']
            media_url = story['media_url']
            pub_date = story['publish_date']
            stories_id = story['stories_id']
            guid = story['guid']
            processed_stories_id = story['processed_stories_id']

            # hash the guid to get unique id
            hash_obj = hashlib.blake2b(digest_size=20)
            hash_obj.update(guid.encode('utf-8'))
            hashed_id = hash_obj.hexdigest()

            # get authors from the story with newspaper
            author = article.authors

            response = Story(hashed_id, title, author, media_name, media_url, pub_date, stories_id, guid,
                             processed_stories_id, text)
    else:
        status = 'fail'
        return Response(response, status)

    if save_format == 'txt':
        txt_file_name = ''.join([hashed_id, '.txt'])
        save_as_txt(''.join(['output_2021/', cur_topic]), txt_file_name, text)
    elif save_format == 'json':
        json_file_name = ''.join([hashed_id, '.json'])
        save_as_json(''.join(['output_2021/', cur_topic]), json_file_name, response)
    elif save_format == 'csv':
        csv_file_name = ''.join([hashed_id, '.csv'])
        save_as_csv(''.join(['output_2021/', cur_topic]), csv_file_name, response)

    return Response(response, status)


def get_many_articles(cur_topics, stories, save_format='json'):
    responses = []
    counter = Counter()
    workers = min(MAX_WORKERS, len(stories))

    with futures.ThreadPoolExecutor(workers) as executor:
        to_do_map = {}
        for story in stories:
            future = executor.submit(get_one_article, story, cur_topics, save_format)
            to_do_map[future] = story
        done_iter = futures.as_completed(to_do_map)

        for future in tqdm(done_iter, total=len(stories), ascii=True):
            try:
                res = future.result()
            except newspaper.ArticleException as article_exc:
                print(article_exc)
                get_many_status = 'fail'
            except requests.exceptions.HTTPError as exc:
                get_many_status = 'fail'
                error_msg = 'HTTP error {res.status_code} - {res.reason}'
                error_msg = error_msg.format(res=exc.response)
                print(error_msg)
            except requests.exceptions.ConnectionError as exc:
                get_many_status = 'fail'
                print('Connection error')
            else:
                get_many_status = res.status
                responses.append(res.response)

            counter[get_many_status] += 1

    return Responses(responses, counter)


if __name__ == '__main__':

    # SET YOUR API KEYS IN THE TXT FILE !!!
    apis = get_list_of_APIs('api_key.txt')
    api_gen = (api for api in apis)
    mc = mediacloud.api.MediaCloud(next(api_gen)[0])  # call the generator

    # SET YOUR QUERY TOPICS HERE !!!
    query_topics = ["abortion", "gay marriage", "death penalty", "euthanasia", "border wall", "immigration ban",
                    "sanctuary cities", "muslim surveillance", "no-fly list gun control", "drug policy",
                    "net neutrality",
                    "affirmative action", "social media regulation", "social security", "obamacare", "marijuana",
                    "climate change",
                    "paris climate agreement", "fracking", "minimum wage", "corporate tax", "equal pay", "welfare",
                    "NAFTA", "tariffs", "china tariffs", "federal reserve", "farm subsidies", "bitcoin",
                    "electoral college",
                    "voter fraud", "campaign finance", "lobbyists", "military spending", "united nations", "torture",
                    "NATO",
                    "israel", "North Korea", "Ukraine", "Russia", "terrorism", "foreign aid", "drones", "Cuba",
                    "student loans",
                    "common core", "private prisons", "mandatory minimum prison sentences", "mandatory vaccinations",
                    "GMO labels",
                    "gerrymandering"]

    # SET YOUR PERIOD HERE !!!
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 24)

    period = mc.dates_as_query_clause(start_date, end_date)

    for topics in query_topics:
        # for period in periods:
        for media in Media:
            cur_media_id = media.value
            media_id = ''.join(["media_id:", str(cur_media_id)])
            query = ''.join([topics, ' AND ', media_id])
            res_stories = stories_about_topic(api_gen,
                                              mc,
                                              query,
                                              period,
                                              fetch_size=50,
                                              limit=50)
            print("We have fetched {} stories from {} about {}".format(len(res_stories), media.name, topics))
            if len(res_stories) != 0:
                story_responses = get_many_articles(topics, res_stories, save_format='json')
                print("Finished! {} success, and {} failure".format(story_responses.count['success'],
                                                                    story_responses.count['fail']))
                print('*' * 40)
                print()
