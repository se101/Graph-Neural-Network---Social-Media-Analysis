import random
import pandas as pd
import numpy as np
import json
import torch
import pickle
import os
from torch_geometric.data import HeteroData
import regex as re
from datetime import datetime
import torch
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

paths = ["/scratch/dipanjan/Twitter_Analysis/Combined_tweet/Combined_Tweets/","/scratch/dipanjan/Twitter_Analysis/Combined_tweet/Combined_Retweets/"]
hashtagcode_path = "List_of_Daily_Hashtags_Coded.csv"
embedding_name_path = "Naming_dict_tweets_embeddings.csv"
embedding_name_pathr = "Naming_dict_retweets_embeddings.csv"
embed_path = '/scratch/dipanjan/Twitter_Analysis/Embeddings/Combined_Tweet_Embeddings/Embeddings/'
rembed_path = '/scratch/dipanjan/Twitter_Analysis/Embeddings/Combined_Tweet_Embeddings/'


def initialize_hashtagdict_rdlist():
    rt_df = pd.read_csv(hashtagcode_path, dtype=object)
    rt_df = rt_df.drop(rt_df.columns[[2, 3]], axis=1)
    rt_df = rt_df.dropna(subset=['Final Code'])
    readlist = rt_df.Hashtag.values.tolist()
    rdlist = []
    for it in readlist:
        rdlist.append(it)
    coding_to_num = {'E': 0, 'G': 1, 'I': 2, 'M': 3, 'S': 4}

    coddf = rt_df.replace({"Final Code": coding_to_num})
    hashtag_dict = coddf.set_index('Hashtag').to_dict()['Final Code']
    rdlist = rdlist
    ret_hr = dict()
    ret_hr['hashtag_dict'] = hashtag_dict
    ret_hr['rdlist'] = rdlist
    return ret_hr


def remove_prefix(input_string, prefix):
    if input_string.startswith(prefix):
        line_new = input_string[len(prefix):]
        return line_new
    return input_string


def datetimeret(inp_date):
    str_date = inp_date[60:70]  # 2017-08-24 18:56:46
    # date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')
    ret_date = datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')
    return ret_date


def initialize_ecd_filename_dict():
    df = pd.read_csv(embedding_name_path, dtype=object)
    dict1 = pd.Series(df.Embedding.values, index=df.encode_Hashtag).to_dict()
    ecd_filename_dict = {}
    for x, y in dict1.items():
        x = remove_prefix(x, 'encode_')
        x = x+'_data.csv'
        ecd_filename_dict[x] = y
    return ecd_filename_dict


def initialize_ecd_filename_dictr():
    df = pd.read_csv(embedding_name_pathr, dtype=object)
    dict1 = pd.Series(df.Embeddings.values,
                      index=df.encode_re_Hashtag).to_dict()
    ecd_filename_dictr = {}
    for x, y in dict1.items():
        x = remove_prefix(x, 'encode_re_')
        x = x+'_rt_data.csv'
        ecd_filename_dictr[x] = y
    return ecd_filename_dictr


def build_nodes():
    ecd_filename_dict = initialize_ecd_filename_dict()
    ecd_filename_dictr = initialize_ecd_filename_dictr()
    ret_hr = initialize_hashtagdict_rdlist()
    hashtag_dict = ret_hr['hashtag_dict']
    rdlist = ret_hr['rdlist']
    print(len(rdlist))
    print(rdlist)
    mapping_users = {}
    user_features = []
    tweet_hashtag = []
    train_mask = []
    test_mask = []
    mapping_tweets = {}
    cou_tweets = 0
    cou_users = 0
    tweet_features = []
    hashtag_features = []
    mapping_hashtag = {}
    cou_hashtags = 0
    count_missing_embedding = 0
    incorrect_embedding = 0
    # all files in a csv and give index
    # code to give all files of a folder an index
    # code to map each file name to a code
    # iterrate based on that order
    # x = [] #filename - csv encode it      x.append(model.encode(filename))
    # y = [] #y.append(map[filename] coding])
    filelist = []
    missing_embedding = []
    t = ""
    nonexistfiles = []
    for path in paths:
        for filename in rdlist:
            if (path == paths[1]):
                filename1 = filename+'_rt_data.csv'
                continue
            else:
                filename1 = filename+'_data.csv'

            if os.path.exists(path+filename1) == False:
                nonexistfiles.append(filename1)
                continue
            if t != "":
                if t == filename:
                    t = ""
                else:
                    continue

            df = pd.read_csv(path+filename1, dtype=object, lineterminator='\n')

            if ('user_tweet_count\r' in df.columns):
                df.rename(
                    columns={'user_tweet_count\r': 'user_tweet_count'}, inplace=True)
            if ('user_tweet_count' not in df.columns):
                filelist.append(filename1)
                continue

            df = df.dropna(subset=['tweet_id',
                                   'author_id',
                                   'mentions',
                                   'retweet_count',
                                   'like_count',
                                   'quote_count',
                                   'referenced_tweets',
                                   'user_followers_count',
                                   'user_following_count',
                                   'user_listed_count',
                                   'user_tweet_count'])
            if (len(df) == 0):
                filelist.append(filename1)
                continue
            embeddings = []
            if filename1 not in ecd_filename_dict.keys() and filename1 not in ecd_filename_dictr.keys():
                missing_embedding.append(filename1)
                continue
            elif filename1 in ecd_filename_dict.keys():
                embed_filename = ecd_filename_dict[filename1]
            elif filename1 in ecd_filename_dictr.keys():
                embed_filename = ecd_filename_dictr[filename1]

            try:
                if filename1 in ecd_filename_dict.keys():
                    embed_filename = embed_path+embed_filename
                    embeddings = pd.read_csv(
                        embed_filename, dtype=object, lineterminator='\n')
                elif filename1 in ecd_filename_dictr.keys():
                    embed_filename = rembed_path+embed_filename
                    embeddings = pd.read_csv(
                        embed_filename, dtype=object, lineterminator='\n')
            except Exception as error:
                print("ERROR:", error)
                continue

            dict_embed = {}
            for i in range(len(embeddings)):
                dict_embed[embeddings['tweet_id'][i]] = re.findall(
                    r"[-]?\d+.\d+[e]?[-]?\d+", embeddings['Embeddings'][i])
            h = []
            tempfname = filename
            findex = rdlist.index(filename)
            if findex not in mapping_hashtag:
                mapping_hashtag[findex] = cou_hashtags
                hashtag_features.append(hashtag_dict[rdlist[findex]])
                rand_test = random.randint(0, 100000)
                train_mask.append(rand_test%2)
                test_mask.append((rand_test+1)%2)
                cou_hashtags = cou_hashtags + 1

            a = []
            df['referenced_tweets'] = df['referenced_tweets'].replace(
                {'\'': '"'}, regex=True)
            df['mentions'] = df['mentions'].replace({'\'': '"'}, regex=True)
            f = 0
            #
            for i in range(0, len(df)):
                try:
                    if (type(df["mentions"][i]) != str):
                        df["mentions"][i] = []
                    else:
                        df["mentions"][i] = json.loads(df["mentions"][i])
                    l = []
                    if df["author_id"][i] not in mapping_users:
                        mapping_users[int(
                            float(df["author_id"][i]))] = cou_users
                        l.append(int(float(df["user_followers_count"][i])))
                        l.append(int(float(df["user_following_count"][i])))
                        # l.append(app_date)
                        l.append(int(float(df["user_listed_count"][i])))
                        l.append(int(float(df["user_tweet_count"][i])))
                        user_features.append(l)
                        cou_users = cou_users+1

                    if (type(df["referenced_tweets"][i]) != str):
                        df["referenced_tweets"][i] = []
                    else:
                        df["referenced_tweets"][i] = json.loads(
                            df["referenced_tweets"][i])

                    ttype = 1

                    if (len(df["referenced_tweets"][i]) > 0):
                        if (df["referenced_tweets"][i][0]["type"] == "quoted"):
                            a.append('q')
                            ttype = 1
                            # print('q')
                        elif (df["referenced_tweets"][i][0]["type"] == "replied_to"):
                            a.append('c')
                            ttype = 2
                            # print('c')
                        elif (df["referenced_tweets"][i][0]["type"] == "retweeted"):
                            a.append('r')
                            ttype = 3
                            # print('r')
                    else:
                        a.append('o')
                        ttype = 4
                        # print('o')
                    if ((int(float(df["tweet_id"][i])) not in mapping_tweets) and (a[i] == 'q' or a[i] == 'o' or a[i] == 'c' or a[i] == 'r')):
                        dic = np.zeros(768)
                        if (df["tweet_id"][i] in dict_embed):
                            dic = np.array(dict_embed[df["tweet_id"][i]])
                        else:
                            count_missing_embedding += 1
                        dic1 = dic.astype(np.float)
                        if (dic1.size < 768):
                            incorrect_embedding += 1
                        while (dic1.size < 768):
                            dic1 = np.append(dic1, [0])
                        dic1 = np.append(dic1, [ttype, float(df["retweet_count"][i]), float(
                            df["like_count"][i]), float(df["quote_count"][i])])
                        v = dic1.tolist()
                        tweet_features.append(v)
                        tweet_hashtag.append(hashtag_dict[rdlist[findex]])
                        mapping_tweets[int(
                            float(df["tweet_id"][i]))] = cou_tweets

                        cou_tweets = cou_tweets+1
                except Exception as error:
                    print("ERROR:", error)
                    f = f+1

    ret_nodes = dict()
    ret_nodes['tweet_hashtag'] = tweet_hashtag
    ret_nodes['mapping_users'] = mapping_users
    ret_nodes['test_mask'] = test_mask
    ret_nodes['train_mask'] = train_mask
    ret_nodes['user_features'] = user_features
    ret_nodes['cou_users'] = cou_users
    ret_nodes['mapping_tweets'] = mapping_tweets
    ret_nodes['tweet_features'] = tweet_features

    ret_nodes['cou_tweets'] = cou_tweets
    ret_nodes['mapping_hashtag'] = mapping_hashtag
    ret_nodes['hashtag_features'] = hashtag_features
    ret_nodes['cou_hashtags'] = cou_hashtags

    pickle.dump(ret_nodes, open("ret_nodes.pickle", "wb"), protocol=4)
    print("incorrect embeddings: ", incorrect_embedding)
    print("missing embeddings: ", count_missing_embedding)
    return ret_nodes


def build_edges():
    nonexistfiles = []
    missing_embedding = []
    ecd_filename_dict = initialize_ecd_filename_dict()
    ecd_filename_dictr = initialize_ecd_filename_dictr()
    ret_hr = initialize_hashtagdict_rdlist()
    hashtag_dict = ret_hr['hashtag_dict']
    rdlist = ret_hr['rdlist']

    t = ""

    with open('ret_nodes.pickle', 'rb') as handle:
        ret_nodes = pickle.load(handle)

    mapping_tweets = ret_nodes['mapping_tweets']
    mapping_users = ret_nodes['mapping_users']
    mapping_hashtag = ret_nodes['mapping_hashtag']

    edge_user_qtweet_src = []
    edge_user_qtweet_dst = []
    label_user_qtweet = []

    edge_user_otweet_src = []
    edge_user_otweet_dst = []
    label_user_otweet = []

    edge_user_ctweet_src = []
    edge_user_ctweet_dst = []
    label_user_ctweet = []

    edge_user_rtweet_src = []
    edge_user_rtweet_dst = []
    label_user_rtweet = []

    edge_rtweet_tweet_src = []
    edge_rtweet_tweet_dst = []
    label_rtweet_tweet = []

    edge_mention_src = []  # tweet to user
    edge_mention_dst = []
    label_mention = []

    edge_qtweet_tweet_src = []
    edge_qtweet_tweet_dst = []
    label_qtweet_tweet = []

    edge_ctweet_tweet_src = []
    edge_ctweet_tweet_dst = []
    label_ctweet_tweet = []

    to_scrape_tweets = []
    to_scrape_users = []

    edge_hashtag_tweet_src = []
    edge_hashtag_tweet_dst = []
    label_hashtag_tweet = []

    for path in paths:
        for filename in rdlist:
            if (path == paths[1]):
                filename1 = filename+'_rt_data.csv'
                continue
            else:
                filename1 = filename+'_data.csv'

            if os.path.exists(path+filename1) == False:
                nonexistfiles.append(filename1)
                continue
            if t != "":
                if t == filename:
                    t = ""
                else:
                    continue

            df = pd.read_csv(path+filename1, dtype=object, lineterminator='\n')
            if ('user_tweet_count\r' in df.columns):
                df.rename(
                    columns={'user_tweet_count\r': 'user_tweet_count'}, inplace=True)
            if ('user_tweet_count' not in df.columns):
                continue
            if filename1 not in ecd_filename_dict.keys() and filename1 not in ecd_filename_dictr.keys():
                missing_embedding.append(filename1)
                continue
            df = df.dropna(subset=['tweet_id',
                                   'author_id',
                                   'mentions',
                                   'retweet_count',
                                   'like_count',
                                   'quote_count',
                                   'referenced_tweets',
                                   'user_followers_count',
                                   'user_following_count',
                                   'user_listed_count',
                                   'user_tweet_count'])
            df['mentions'] = df['mentions'].replace({'\'': '"'}, regex=True)

            for i in range(0, len(df)):
                if (type(df["mentions"][i]) != str):
                    df["mentions"][i] = []
                else:
                    df["mentions"][i] = json.loads(df["mentions"][i])

            cou = 0

            df['referenced_tweets'] = df['referenced_tweets'].replace(
                {'\'': '"'}, regex=True)

            # for i in range(0,len(df)):
            #    df["referenced_tweets"][i]=json.loads(df["referenced_tweets"][i])

            for i in range(0, len(df)):
                if (type(df["referenced_tweets"][i]) != str):
                    df["referenced_tweets"][i] = []
                else:
                    df["referenced_tweets"][i] = json.loads(
                        df["referenced_tweets"][i])

            a = []
            for i in range(0, len(df)):
                if (len(df["referenced_tweets"][i]) > 0):
                    if (df["referenced_tweets"][i][0]["type"] == "quoted"):
                        a.append('q')
                    elif (df["referenced_tweets"][i][0]["type"] == "replied_to"):
                        a.append('c')
                    elif (df["referenced_tweets"][i][0]["type"] == "retweeted"):
                        a.append('r')
                else:
                    a.append('o')
            df["tweet_type"] = a

            tempfname = filename
            findex = rdlist.index(filename)

            for i in range(0, len(df)):
                if (df["tweet_type"][i] == 'r'):
                    label_user_rtweet.append(1)
                    edge_user_rtweet_src.append(
                        mapping_users[float(df["author_id"][i])])
                    edge_user_rtweet_dst.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_dst.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_src.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    if df["referenced_tweets"][i][0]['id'] in mapping_tweets:
                        label_rtweet_tweet.append(1)
                        edge_rtweet_tweet_src.append(
                            mapping_tweets[float(df["tweet_id"][i])])
                        edge_rtweet_tweet_dst.append(
                            mapping_tweets[int(df["referenced_tweets"][i][0]['id'])])
                    else:
                        to_scrape_tweets.append(
                            df["referenced_tweets"][i][0]['id'])

                elif df["tweet_type"][i] == 'q':
                    label_user_qtweet.append(1)
                    edge_user_qtweet_src.append(
                        mapping_users[float(df["author_id"][i])])
                    edge_user_qtweet_dst.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_dst.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_src.append(
                        mapping_tweets[float(df["tweet_id"][i])])
                    if int(df["referenced_tweets"][i][0]['id']) in mapping_tweets:
                        label_qtweet_tweet.append(1)
                        edge_qtweet_tweet_src.append(
                            mapping_tweets[float(df["tweet_id"][i])])
                        edge_qtweet_tweet_dst.append(
                            mapping_tweets[int(df["referenced_tweets"][i][0]['id'])])
                    else:
                        to_scrape_tweets.append(
                            df["referenced_tweets"][i][0]['id'])

                    for j in range(1, len(df["mentions"][i])):
                        if float(df["mentions"][i][j]["id"]) in mapping_users:
                            label_mention.append(1)
                            edge_mention_src.append(
                                mapping_tweets[float(df["tweet_id"][i])])
                            edge_mention_dst.append(
                                mapping_users[float(df["mentions"][i][j]["id"])])
                        else:
                            to_scrape_users.append(df["mentions"][i][j]["id"])

                elif df["tweet_type"][i] == 'o':
                    label_user_otweet.append(1)
                    edge_user_otweet_src.append(
                        mapping_users[float(df["author_id"][i])])
                    edge_user_otweet_dst.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_dst.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_src.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    for j in range(1, len(df["mentions"][i])):
                        if float(df["mentions"][i][j]["id"]) in mapping_users:
                            label_mention.append(1)
                            edge_mention_src.append(
                                mapping_tweets[float(df["tweet_id"][i])])
                            edge_mention_dst.append(
                                mapping_users[float(df["mentions"][i][j]["id"])])
                        else:
                            to_scrape_users.append(df["mentions"][i][j]["id"])

                elif df["tweet_type"][i] == 'c':
                    label_user_ctweet.append(1)
                    edge_user_ctweet_src.append(
                        mapping_users[float(df["author_id"][i])])
                    edge_user_ctweet_dst.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_dst.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_src.append(
                        mapping_tweets[float(df["tweet_id"][i])])

                    if float(df["referenced_tweets"][i][0]['id']) in mapping_tweets:
                        label_ctweet_tweet.append(1)
                        # CHANGED FROM edge_hashtag_tweet_src
                        edge_ctweet_tweet_src.append(mapping_hashtag[findex])
                        edge_ctweet_tweet_dst.append(
                            mapping_tweets[float(df["tweet_id"][i])])
                    else:
                        to_scrape_tweets.append(
                            df["referenced_tweets"][i][0]['id'])

                    for j in range(1, len(df["mentions"][i])):
                        if float(df["mentions"][i][j]["id"]) in mapping_users:
                            label_mention.append(1)
                            edge_mention_src.append(
                                mapping_tweets[float(df["tweet_id"][i])])
                            edge_mention_dst.append(
                                mapping_users[float(df["mentions"][i][j]["id"])])
                        else:
                            to_scrape_users.append(df["mentions"][i][j]["id"])
                else:
                    print("Nothing happened: "+str(i))

    ret_edges = dict()

    ret_edges['edge_user_qtweet_src'] = edge_user_qtweet_src
    ret_edges['edge_user_qtweet_dst'] = edge_user_qtweet_dst
    ret_edges['label_user_qtweet'] = label_user_qtweet

    ret_edges['edge_user_otweet_src'] = edge_user_otweet_src
    ret_edges['edge_user_otweet_dst'] = edge_user_otweet_dst
    ret_edges['label_user_otweet'] = label_user_otweet

    ret_edges['edge_user_ctweet_src'] = edge_user_ctweet_src
    ret_edges['edge_user_ctweet_dst'] = edge_user_ctweet_dst
    ret_edges['label_user_ctweet'] = label_user_ctweet

    ret_edges['edge_user_rtweet_src'] = edge_user_rtweet_src
    ret_edges['edge_user_rtweet_dst'] = edge_user_rtweet_dst
    ret_edges['label_user_rtweet'] = label_user_rtweet

    ret_edges['edge_qtweet_tweet_src'] = edge_qtweet_tweet_src
    ret_edges['edge_qtweet_tweet_dst'] = edge_qtweet_tweet_dst
    ret_edges['label_qtweet_tweet'] = label_qtweet_tweet

    ret_edges['edge_ctweet_tweet_src'] = edge_ctweet_tweet_src
    ret_edges['edge_ctweet_tweet_dst'] = edge_ctweet_tweet_dst
    ret_edges['label_ctweet_tweet'] = label_ctweet_tweet

    ret_edges['edge_mention_src'] = edge_mention_src
    ret_edges['edge_mention_dst'] = edge_mention_dst
    ret_edges['label_mention'] = label_mention

    ret_edges['edge_hashtag_tweet_src'] = edge_hashtag_tweet_src
    ret_edges['edge_hashtag_tweet_dst'] = edge_hashtag_tweet_dst
    ret_edges['label_hashtag_tweet'] = label_hashtag_tweet

    ret_edges['to_scrape_tweets'] = to_scrape_tweets
    ret_edges['to_scrape_users'] = to_scrape_users

    pickle.dump(ret_edges, open("ret_edges.pickle", "wb"), protocol=4)

    return ret_edges


def getEdgeAndLabel(edge_from_to_src, edge_from_to_dst, label_from_to):
    edge_from_to_src = np.array(edge_from_to_src)
    edge_from_to_src = edge_from_to_src.reshape((edge_from_to_src.shape[0], 1))

    edge_from_to_dst = np.array(edge_from_to_dst)
    edge_from_to_dst = edge_from_to_dst.reshape((edge_from_to_dst.shape[0], 1))

    edge_from_to = [edge_from_to_src, edge_from_to_dst]
    edge_from_to = np.array(edge_from_to)
    edge_from_to = edge_from_to.reshape(
        (edge_from_to.shape[0], edge_from_to.shape[1]))
    edge_from_to = torch.tensor(edge_from_to.astype('int64'))

    label_from_to = np.array(label_from_to)
    label_from_to = torch.tensor(label_from_to.reshape(
        (label_from_to.shape[0], 1)).astype('int64'))
    return edge_from_to, label_from_to


def build_graph():
    with open('ret_edges.pickle', 'rb') as handle:
        ret_edges = pickle.load(handle)

    edge_user_otweet, label_user_otweet = getEdgeAndLabel(
        ret_edges['edge_user_otweet_src'], ret_edges['edge_user_otweet_dst'], ret_edges['label_user_otweet'])
    edge_user_ctweet, label_user_ctweet = getEdgeAndLabel(
        ret_edges['edge_user_ctweet_src'], ret_edges['edge_user_ctweet_dst'], ret_edges['label_user_ctweet'])
    edge_user_qtweet, label_user_qtweet = getEdgeAndLabel(
        ret_edges['edge_user_qtweet_src'], ret_edges['edge_user_qtweet_dst'], ret_edges['label_user_qtweet'])
    edge_user_rtweet, label_user_rtweet = getEdgeAndLabel(
        ret_edges['edge_user_rtweet_src'], ret_edges['edge_user_rtweet_dst'], ret_edges['label_user_rtweet'])
    edge_ctweet_tweet, label_ctweet_tweet = getEdgeAndLabel(
        ret_edges['edge_ctweet_tweet_src'], ret_edges['edge_ctweet_tweet_dst'], ret_edges['label_ctweet_tweet'])
    edge_qtweet_tweet, label_qtweet_tweet = getEdgeAndLabel(
        ret_edges['edge_qtweet_tweet_src'], ret_edges['edge_qtweet_tweet_dst'], ret_edges['label_qtweet_tweet'])
    edge_mention, label_mention = getEdgeAndLabel(
        ret_edges['edge_mention_src'], ret_edges['edge_mention_dst'], ret_edges['label_mention'])
    edge_hashtag_tweet, label_hashtag_tweet = getEdgeAndLabel(
        ret_edges['edge_hashtag_tweet_src'], ret_edges['edge_hashtag_tweet_dst'], ret_edges['label_hashtag_tweet'])

    data = HeteroData()

    with open('ret_nodes.pickle', 'rb') as handle1:
        ret_nodes = pickle.load(handle1)
    user_features = ret_nodes['user_features']
    tweet_features = ret_nodes['tweet_features']
    tweet_hashtag = ret_nodes['tweet_hashtag']
    hashtag_features = ret_nodes['hashtag_features']
    train_mask = ret_nodes['train_mask']
    test_mask = ret_nodes['test_mask']
    train_mask = torch.tensor(train_mask)
    test_mask = torch.tensor(test_mask)
#    tweet_types=ret_nodes['tweet_types']
#    retweets_count=ret_nodes['retweets_count']
#    tweet_like_count=ret_nodes['tweet_like_count']
#    tweet_quote_count=ret_nodes['tweet_quote_count']
    user_features = torch.tensor(user_features).float()
    # print(tweet_types)
#    tweet_types=torch.tensor(tweet_types)
#    retweets_count=torch.tensor(retweets_count)
#    tweet_like_count=torch.tensor(tweet_like_count)
    tweet_hashtag = torch.tensor(tweet_hashtag).float()
    tweet_features = torch.tensor(tweet_features).float()
#    tweet_quote_count=torch.tensor(tweet_quote_count)
    hashtag_features = torch.tensor(hashtag_features).float()

    data['user'].x = user_features
    data['tweet'].x = tweet_features
    #data['tweet'].y = tweet_hashtag
    # data['tweet'].tweet_type=tweet_types
    # data['tweet'].retweet_count=retweets_count
    # data['tweet'].like_count=tweet_like_count
    # data['tweet'].quote_count=tweet_quote_count

    data['hashtag'].y=hashtag_features
    data['hashtag'].train_mask = train_mask
    data['hashtag'].test_mask = test_mask

    data['user', 'post_original', 'tweet'].edge_index = edge_user_otweet
    # data['user','post_original','tweet'].edge_label=label_user_otweet

    data['user', 'post_quote', 'tweet'].edge_index = edge_user_qtweet
    # data['user','post_quote','tweet'].edge_label=label_user_qtweet

    data['user', 'post_reply', 'tweet'].edge_index = edge_user_ctweet
    # data['user','post_reply','tweet'].edge_label=label_user_ctweet

    # data['user','post_retweet','tweet'].edge_index=edge_user_rtweet
    # data['user','post_retweet','tweet'].edge_label=label_user_rtweet

    data['tweet', 'quotes', 'tweet'].edge_index = edge_qtweet_tweet
    # data['tweet','quotes','tweet'].edge_label=label_qtweet_tweet

    data['tweet', 'replies', 'tweet'].edge_index = edge_ctweet_tweet
    # data['tweet','replies','tweet'].edge_label=label_ctweet_tweet

    data['tweet', 'mentions', 'user'].edge_index = edge_mention
    # data['tweet','mentions','user'].edge_label=label_mention

    data['tweet','hashtagedge','hashtag'].edge_index=edge_hashtag_tweet
    # data['hashtag','hashtagedge','tweet'].edge_label=label_hashtag_tweet
    print(data)
    pickle.dump(data, open("GRAPH_HASHTAG_LARGE.pickle", "wb"), protocol=4)
    return data

# NODES = build_nodes()
# print("NODES FORMATION COMPLETE")
# EDGES = build_edges()
# print("EDGE FORMATION COMPLETE")
# GRAPH = build_graph()
# print("GRAPH FORMATION COMPLETE")


def GNN_Metadata_PreProcessing():
    node_types = ['user', 'tweet']
    edge_types = [
        ('user', 'post_original', 'tweet'),
        ('user', 'post_quote', 'tweet'),
        ('user', 'post_reply', 'tweet'),
        ('tweet', 'quotes', 'tweet'),
        ('tweet', 'replies', 'tweet'),
        ('tweet', 'mentions', 'user')]
    metadata = (node_types, edge_types)
    return metadata


class SAGEConvWithMultipleLinearLayers(torch.nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256, num_hidden_layers=3):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 512)
        self.conv2 = SAGEConv((-1, -1), 256)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        for layer in self.hidden_layers:
            x = layer(x).relu()

        x = self.fc(x)
        return torch.nn.functional.softmax(x, dim=-1)


class GATWithMultipleLinearLayers(torch.nn.Module):
    def __init__(self, hidden_channels=512, out_channels=256, hidden_dim=256, num_hidden_layers=3, num_classes=5):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        for layer in self.hidden_layers:
            x = layer(x).relu()

        x = self.fc(x)
        return torch.nn.functional.softmax(x, dim=-1)


class SimpleGAT(torch.nn.Module):
    def __init__(self, hidden_channels=512, out_channels=5, hidden_dim=256, num_hidden_layers=3, num_classes=5):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin2 = Linear(-1, hidden_dim)
        self.conv3 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin3 = Linear(-1, out_channels)
        self.hidden_layers = torch.nn.ModuleList()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = x.relu()
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


class SimpleSAGEConv(torch.nn.Module):
    def __init__(self, hidden_channels=512, out_channels=5):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class SAGEConvAndGATWithLinearLayers(torch.nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256, num_hidden_layers=4):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 512)
        self.conv2 = SAGEConv((-1, -1), 256)
        self.gat_layers = torch.nn.ModuleList()
        for _ in range(3):
            self.gat_layers.append(
                GATConv((-1, -1), hidden_dim, add_self_loops=False))
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        for layer in self.gat_layers:
            x = layer(x, edge_index).relu()
        for layer in self.hidden_layers:
            x = layer(x).relu()
        x = self.fc(x)
        return torch.nn.functional.softmax(x, dim=-1)


def Model_Preprocessing():
#    model = modeType()
#    metadata = GNN_Metadata_PreProcessing()
#    model = to_hetero(model, metadata)
    NODES = build_nodes()
    print("NODES FORMATION COMPLETE")
    EDGES = build_edges()
    print("EDGE FORMATION COMPLETE")
    build_graph()
    print("GRAPH FORMATION COMPLETE")
#    GRAPH={}
#    with open("GRAPH.pickle",'rb') as f:
#        GRAPH = pickle.load(f)
#    x_dict = GRAPH.x_dict
#    edge_index_dict = GRAPH.edge_index_dict
   # model = modeType()
#    metadata = GNN_Metadata_PreProcessing()
#    model = to_hetero(model, metadata)
    return


def compute_accuracy(predictions, targets):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct_predictions = (predicted_classes == targets).sum().item()
    total_predictions = targets.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy


def train_model():
    model_architectures = [SimpleSAGEConv, SimpleGAT, SAGEConvWithMultipleLinearLayers,
                           GATWithMultipleLinearLayers,  SAGEConvAndGATWithLinearLayers]
    for a in model_architectures:
        print(a)
        GRAPH, x_dict, edge_index_dict, model = Model_Preprocessing(a)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        data = GRAPH.y_dict['tweet']
        num_categories = 5
        for epoch in range(10):
            total_loss = 0
            total_accuracy = 0

            optimizer.zero_grad()

            out = model(x_dict, edge_index_dict)
            out = out['tweet']
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, data.long())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy = compute_accuracy(out, data)
            total_accuracy += accuracy

            print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")

            if (epoch == 9):
                with open("results.txt", "a") as file:
                    file.write(
                        f"model: {a.__name__}, Loss: {loss.item()}, Accuracy: {accuracy}\n")


Model_Preprocessing()
