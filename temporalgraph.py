import pandas as pd
import numpy as np
import json
import torch
import pickle
import os
from torch_geometric.data import HeteroData
import regex as re

paths = ["/home/protests/Documents/hashtag_scrape/Combined_Tweets/","/home/protests/Documents/hashtag_scrape/Combined_Retweets/"]
rdlist = []

def initialize_hashtag_dict():
    rt_df=pd.read_csv(r'List_of_Daily_Hashtags_Coded.csv', dtype=object)
    rt_df = rt_df.drop(rt_df.columns[[2,3]], axis=1)
    rt_df = rt_df.dropna(subset = ['Final Code'])
    df = rt_df.groupby('Final Code')
    readlist = rt_df.Hashtag.values.tolist()
    rdlist = []
    paths = ["/home/protests/Documents/hashtag_scrape/Combined_Tweets/","/home/protests/Documents/hashtag_scrape/Combined_Retweets/"]
    for it in readlist :
            rdlist.append(it)
    coding_to_num = {'E': 0, 'G': 1, 'I': 2, 'M': 3, 'S': 4}
    coddf = rt_df.replace({"Final Code": coding_to_num})
    hashtag_dict = coddf.set_index('Hashtag').to_dict()['Final Code']
    len(hashtag_dict)
    return hashtag_dict

def remove_prefix(input_string, prefix):
    if input_string.startswith(prefix):
        line_new = input_string[len(prefix):]
        return line_new
    return input_string

def initialize_ecd_filename_dict():
    df = pd.read_csv(r"Naming_dict_tweets_embeddings.csv")
    dict1 = pd.Series(df.Embedding.values,index=df.encode_Hashtag).to_dict()
    ecd_filename_dict = {}
    for x, y in dict1.items():
        x = remove_prefix(x, 'encode_')
        x = x+'_data.csv'
        ecd_filename_dict[x]=y
    print(ecd_filename_dict)
    return ecd_filename_dict

def build_nodes():
    ecd_filename_dict = initialize_ecd_filename_dict()
    hashtag_dict = initialize_hashtag_dict()

    mapping_users={}
    user_features=[]
    mapping_tweets={}
    cou_tweets=0
    cou_users=0
    tweet_features=[]
    hashtag_features=[]
    mapping_hashtag={}
    cou_hashtags=0
    #all files in a csv and give index
    #code to give all files of a folder an index
    #code to map each file name to a code
    #iterrate based on that order
    #x = [] #filename - csv encode it      x.append(model.encode(filename))
    #y = [] #y.append(map[filename] coding])
    filelist=[]
    missing_embedding=[]
    t = ""
    nonexistfiles = []
    for path in paths :
        for filename in rdlist:
            if(path==paths[1]):
                filename1=filename+'_rt_data.csv'
            else:
                filename1=filename+'_data.csv'
            
            if os.path.exists(path+filename1) == False :
                nonexistfiles.append(filename1)
                continue
            if t != "":
                if t == filename:
                    t = ""
                else:
                    continue
            
            df=pd.read_csv(path+filename1,dtype=object,lineterminator='\n')
            
            if('user_tweet_count\r' in df.columns):
                df.rename(columns = {'user_tweet_count\r':'user_tweet_count'}, inplace = True)
            if('user_tweet_count' not in df.columns):
                filelist.append(filename1)
                continue
                
            
            df = df.dropna(subset = ['tweet_id', 
                                        'conversation_id',
                                        'tweet_created_at',
                                        'author_id',
                                        'username', 
                                        'mentions',
                                        'reply_count',
                                        'retweet_count',
                                        'like_count',
                                        'quote_count',
                                        'referenced_tweets',
                                        'user_followers_count',
                                        'user_following_count',
                                        'user_listed_count',
                                        'user_tweet_count'])
            if(len(df)==0):
                filelist.append(filename1)
                continue
            if filename1 not in ecd_filename_dict.keys():
                missing_embedding.append(filename1)
                continue
            embed_filename = ecd_filename_dict[filename1]
            embeddings=[]
            try:
                
                embed_filename = '/home/protests/Documents/GNN/Tweet_Embeddings/Combined_Tweet_Embeddings/Embeddings/'+embed_filename
                embeddings=pd.read_csv(embed_filename,dtype=object,lineterminator='\n')
            except:
                print(embed_filename)
                continue
            
            dict_embed = {}
            for i in range(len(embeddings)):
                dict_embed[embeddings['tweet_id'][i]] = re.findall(r"[-]?\d+.\d+[e]?[-]?\d+", embeddings['Embeddings'][i])
                
            h=[]
            tempfname = filename
            findex = rdlist.index(filename)
            if findex not in mapping_hashtag :
                mapping_hashtag[findex] = cou_hashtags
                hashtag_features.append(hashtag_dict[rdlist[findex]])
                cou_hashtags = cou_hashtags + 1


            a=[]
            df['referenced_tweets'] = df['referenced_tweets'].replace({'\'': '"'}, regex=True)        
            df['mentions'] = df['mentions'].replace({'\'': '"'}, regex=True)
            f = 0
            #
            for i in range(0,len(df)):
                try:
                    if(type(df["mentions"][i])!=str):
                        df["mentions"][i]=[]
                    else:
                        df["mentions"][i]=json.loads(df["mentions"][i])
                    l=[]
                    if df["author_id"][i] not in mapping_users:
                        mapping_users[df["author_id"][i]]=cou_users
                        l.append(int(float(df["user_followers_count"][i])))
                        l.append(int(float(df["user_following_count"][i])))

                        l.append(int(float(df["user_listed_count"][i])))
                        l.append(int(float(df["user_tweet_count"][i])))
                        user_features.append(l)
                        cou_users=cou_users+1

                    if(type(df["referenced_tweets"][i])!=str):
                        df["referenced_tweets"][i]=[]
                    else:
                        df["referenced_tweets"][i]=json.loads(df["referenced_tweets"][i])


                    if(len(df["referenced_tweets"][i])>0):
                        if(df["referenced_tweets"][i][0]["type"]=="quoted"):
                            a.append('q')
                            #print('q')
                        elif(df["referenced_tweets"][i][0]["type"]=="replied_to"):
                            a.append('c')
                            #print('c')
                        elif(df["referenced_tweets"][i][0]["type"]=="retweeted"):
                            a.append('r')
                            #print('r')
                    else:
                            a.append('o')
                            #print('o')
                    l=[]
                    if ((df["tweet_id"][i] not in mapping_tweets) and (a[i]=='q' or a[i]=='o' or a[i]=='c')):
                        #print(i, "xd")
                        dic = np.array(dict_embed[df["tweet_id"][i]])
                        dic1 = dic.astype(np.float)
                        if i == 1:
                            ace = dic1[0]+dic1[1]
                            print(ace)
                        mapping_tweets[df["tweet_id"][i]]=cou_tweets
                        l.append(int(float(df["retweet_count"][i])))
                        l.append(int(float(df["like_count"][i])))
                        l.append(int(float(df["quote_count"][i])))
                        s=""
                        l.append(dic1)
                        tweet_features.append(l)
                        cou_tweets=cou_tweets+1   
                except:
                    f = f+1
            try:
                df["tweet_type"]=a
            except:
                f = f
            print(f)

    ret_nodes = dict();
    
    ret_nodes['mapping_users'] = mapping_users
    ret_nodes['user_features'] = user_features
    ret_nodes['cou_users'] = cou_users

    ret_nodes['mapping_tweets'] = mapping_tweets
    ret_nodes['tweet_features'] = tweet_features
    ret_nodes['cou_tweets'] = cou_tweets

    ret_nodes['mapping_hashtag'] = mapping_hashtag
    ret_nodes['hashtag_features'] = hashtag_features
    ret_nodes['cou_hashtags'] = cou_hashtags

    with open('ret_nodes.pickle', 'wb') as fp:
        pickle.dump(ret_nodes, fp)

    return ret_nodes
    

def build_edges():
    nonexistfiles=[]
    missing_embedding=[]
    ecd_filename_dict = initialize_ecd_filename_dict()

    with open('ret_nodes.pickle', 'rb') as handle:
        ret_nodes = pickle.load(handle)

    mapping_tweets = ret_nodes['mapping_tweets']
    mapping_users = ret_nodes['mapping_users']
    mapping_hashtag = ret_nodes['mapping_hastag']

    edge_user_qtweet_src=[]
    edge_user_qtweet_dst=[]
    label_user_qtweet=[]
    edge_user_otweet_src=[]
    edge_user_otweet_dst=[]
    label_user_otweet=[]
    edge_user_ctweet_src=[]
    edge_user_ctweet_dst=[]
    label_user_ctweet=[]
    edge_user_rtweet_src=[]
    edge_user_rtweet_dst=[]
    label_user_rtweet=[]

    edge_qtweet_tweet_src=[]
    edge_qtweet_tweet_dst=[]
    label_qtweet_tweet=[]
    edge_ctweet_tweet_src=[]
    edge_ctweet_tweet_dst=[]
    label_ctweet_tweet=[]

    edge_tweet_user_src=[]
    edge_tweet_user_dst=[]
    label_tweet_user=[]
    to_scrape_tweets=[]
    to_scrape_users=[]

    edge_hashtag_tweet_src=[]
    edge_hashtag_tweet_dst=[]
    label_hashtag_tweet = []

    for path in paths :
        for filename in rdlist :
            if(path=='./Combined_Retweets/'):
                filename1=filename+'_data.csv'
            else:
                filename1=filename+'_rt_data.csv'
            if os.path.exists(path+filename1) == False :
                nonexistfiles.append(filename1)
                continue
            if t != "":
                if t == filename:
                    t = ""
                else:
                    continue

            df=pd.read_csv(path+filename1,dtype=object,lineterminator='\n')
            if filename1 not in ecd_filename_dict.keys():
                missing_embedding.append(filename1)
                continue
            df = df.dropna(subset = ['tweet_id', 
                                        'conversation_id',
                                        'tweet_created_at',
                                        'author_id',
                                        'username', 
                                        'mentions',
                                        'reply_count',
                                        'retweet_count',
                                        'like_count',
                                        'quote_count',
                                        'referenced_tweets',
                                        'user_followers_count',
                                        'user_following_count',
                                        'user_listed_count',
                                        'user_tweet_count'])
            df['mentions'] = df['mentions'].replace({'\'': '"'}, regex=True)

            for i in range(0,len(df)):    
                if(type(df["mentions"][i])!=str):
                    df["mentions"][i]=[]
                else:
                    df["mentions"][i]=json.loads(df["mentions"][i])
                
            cou=0
            
            df['referenced_tweets'] = df['referenced_tweets'].replace({'\'': '"'}, regex=True)   
            
            #for i in range(0,len(df)):
            #    df["referenced_tweets"][i]=json.loads(df["referenced_tweets"][i])
                
            for i in range(0,len(df)):    
                if(type(df["referenced_tweets"][i])!=str):
                    df["referenced_tweets"][i]=[]
                else:
                    df["referenced_tweets"][i]=json.loads(df["referenced_tweets"][i])
            
            a=[]
            for i in range(0,len(df)):
                if(len(df["referenced_tweets"][i])>0):
                    if(df["referenced_tweets"][i][0]["type"]=="quoted"):
                        a.append('q')
                    elif(df["referenced_tweets"][i][0]["type"]=="replied_to"):
                        a.append('c')
                    elif(df["referenced_tweets"][i][0]["type"]=="retweeted"):
                        a.append('r')
                else:
                        a.append('o')
            df["tweet_type"]=a
            
            tempfname = filename
            findex = rdlist.index(filename)

            for i in range(0,len(df)):
                print(i)
                if(df["tweet_type"][i]=='r'):
                    if df["referenced_tweets"][i][0]['id']  in mapping_tweets:
                        label_user_rtweet.append(1)
                        edge_user_rtweet_src.append(mapping_users[df["author_id"][i]])
                        edge_user_rtweet_dst.append(mapping_tweets[df["referenced_tweets"][i][0]['id']])

                        label_hashtag_tweet.append(1)
                        edge_hashtag_tweet_src.append(mapping_hashtag[findex])
                        print("edges_hashtag")
                        edge_hashtag_tweet_dst.append(mapping_tweets[df["referenced_tweets"][i][0]['id']])
                    else:
                        to_scrape_tweets.append(df["referenced_tweets"][i][0]['id'] )  
                    
                elif df["tweet_type"][i]=='q':
                    if df["referenced_tweets"][i][0]['id']  in mapping_tweets:
                        label_qtweet_tweet.append(1)
                        edge_qtweet_tweet_src.append(mapping_tweets[df["tweet_id"][i]])
                        edge_qtweet_tweet_dst.append(mapping_tweets[df["referenced_tweets"][i][0]['id']])

                        label_hashtag_tweet.append(1)
                        edge_hashtag_tweet_src.append(mapping_hashtag[findex])
                        edge_hashtag_tweet_dst.append(mapping_tweets[df["referenced_tweets"][i][0]['id']])
                    else:
                        to_scrape_tweets.append(df["referenced_tweets"][i][0]['id'] )    

                    label_user_qtweet.append(1)
                    edge_user_qtweet_src.append(mapping_users[df["author_id"][i]])
                    edge_user_qtweet_dst.append(mapping_tweets[df["tweet_id"][i]])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_src.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_dst.append(mapping_tweets[df["tweet_id"][i]])

                    for j in range(1,len(df["mentions"][i])):
                        if df["mentions"][i][j]["id"] in mapping_users:
                            label_tweet_user.append(1)
                            edge_tweet_user_src.append(mapping_tweets[df["tweet_id"][i]])
                            edge_tweet_user_dst.append(mapping_users[df["mentions"][i][j]["id"]])
                        else:
                            to_scrape_users.append(df["mentions"][i][j]["id"])    
                elif df["tweet_type"][i]=='o':
                    label_user_otweet.append(1)
                    edge_user_otweet_src.append(mapping_users[df["author_id"][i]])
                    edge_user_otweet_dst.append(mapping_tweets[df["tweet_id"][i]])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_src.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_dst.append(mapping_tweets[df["tweet_id"][i]])

                    for j in range(0,len(df["mentions"][i])):
                        if df["mentions"][i][j]["id"] in mapping_users:            
                            label_tweet_user.append(1)
                            edge_tweet_user_src.append(mapping_tweets[df["tweet_id"][i]])
                            edge_tweet_user_dst.append(df["mentions"][i][j]["id"])
                        else:
                            to_scrape_users.append(df["mentions"][i][j]["id"])
                elif df["tweet_type"][i]=='c':
                    if df["referenced_tweets"][i][0]['id']  in mapping_tweets:
                            label_ctweet_tweet.append(1)
                            edge_ctweet_tweet_src.append(mapping_tweets[df["tweet_id"][i]])
                            edge_ctweet_tweet_dst.append(mapping_tweets[df["referenced_tweets"][i][0]['id']])

                            label_hashtag_tweet.append(1)
                            edge_hashtag_tweet_src.append(mapping_hashtag[findex])
                            edge_hashtag_tweet_dst.append(mapping_tweets[df["tweet_id"][i]])
                    else:
                        to_scrape_tweets.append(df["referenced_tweets"][i][0]['id'] )    

                    label_user_ctweet.append(1)
                    edge_user_ctweet_src.append(mapping_users[df["author_id"][i]])
                    edge_user_ctweet_dst.append(mapping_tweets[df["tweet_id"][i]])

                    label_hashtag_tweet.append(1)
                    edge_hashtag_tweet_src.append(mapping_hashtag[findex])
                    edge_hashtag_tweet_dst.append(mapping_tweets[df["tweet_id"][i]])

                    for j in range(1,len(df["mentions"][i])):
                        if df["mentions"][i][j]["id"] in mapping_users:
                            label_tweet_user.append(1)
                            edge_tweet_user_src.append(mapping_tweets[df["tweet_id"][i]])
                            edge_tweet_user_dst.append(mapping_users[df["mentions"][i][j]["id"]])
                        else:
                            to_scrape_users.append(df["mentions"][i][j]["id"])
                else:
                    print("Nothing happened: "+str(i))

    ret_edges = dict();
    
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

    ret_edges['edge_tweet_user_src'] = edge_tweet_user_src
    ret_edges['edge_tweet_user_dst'] = edge_tweet_user_dst
    ret_edges['label_tweet_user'] = label_tweet_user

    ret_edges['edge_hashtag_tweet_src'] = edge_hashtag_tweet_src
    ret_edges['edge_hashtag_tweet_dst'] = edge_hashtag_tweet_dst
    ret_edges['label_hashtag_tweet'] = label_hashtag_tweet

    ret_edges['to_scrape_tweets'] = to_scrape_tweets
    ret_edges['to_scrape_users'] = to_scrape_users

    with open('ret_edges.pickle', 'wb') as fp:
        pickle.dump(ret_edges, fp)

    return ret_edges

def getEdgeAndLabel(edge_from_to_src,edge_from_to_dst,label_from_to):
    edge_from_to_src=np.array(edge_from_to_src)
    edge_from_to_src=edge_from_to_src.reshape((edge_from_to_src.shape[0],1))

    edge_from_to_dst=np.array(edge_from_to_dst)
    edge_from_to_dst=edge_from_to_dst.reshape((edge_from_to_dst.shape[0],1))

    edge_from_to=[edge_from_to_src,edge_from_to_dst]
    edge_from_to=np.array(edge_from_to)
    edge_from_to=edge_from_to.reshape((edge_from_to.shape[0],edge_from_to.shape[1]))
    edge_from_to = torch.tensor(edge_from_to.astype('int64'))

    label_from_to=np.array(label_from_to)
    label_from_to = torch.tensor(label_from_to.reshape((label_from_to.shape[0],1)).astype('int64'))
    return edge_from_to,label_from_to

def build_graph():
    with open('ret_edges.pickle', 'rb') as handle:
        ret_edges = pickle.load(handle)
    
    edge_user_otweet,label_user_otweet=getEdgeAndLabel(ret_edges['edge_user_otweet_src'],ret_edges['edge_user_otweet_dst'],ret_edges['label_user_otweet'])
    edge_user_ctweet,label_user_ctweet=getEdgeAndLabel(ret_edges['edge_user_ctweet_src'],ret_edges['edge_user_ctweet_dst'],ret_edges['label_user_ctweet'])
    edge_user_qtweet,label_user_qtweet=getEdgeAndLabel(ret_edges['edge_user_qtweet_src'],ret_edges['edge_user_qtweet_dst'],ret_edges['label_user_qtweet'])
    edge_user_rtweet,label_user_rtweet=getEdgeAndLabel(ret_edges['edge_user_rtweet_src'],ret_edges['edge_user_rtweet_dst'],ret_edges['label_user_rtweet'])
    edge_ctweet_tweet,label_ctweet_tweet=getEdgeAndLabel(ret_edges['edge_ctweet_tweet_src'],ret_edges['edge_ctweet_tweet_dst'],ret_edges['label_ctweet_tweet'])
    edge_qtweet_tweet,label_qtweet_tweet=getEdgeAndLabel(ret_edges['edge_qtweet_tweet_src'],ret_edges['edge_qtweet_tweet_dst'],ret_edges['label_qtweet_tweet'])
    edge_tweet_user,label_tweet_user=getEdgeAndLabel(ret_edges['edge_tweet_user_src'],ret_edges['edge_tweet_user_dst'],ret_edges['label_tweet_user'])
    edge_hashtag_tweet,label_hashtag_tweet=getEdgeAndLabel(ret_edges['edge_hashtag_tweet_src'],ret_edges['edge_hashtag_tweet_dst'],ret_edges['label_hashtag_tweet'])

    data = HeteroData()
    
    with open('ret_nodes.pickle', 'rb') as handle1:
        ret_nodes = pickle.load(handle1)

    user_features = ret_nodes['user_features']
    tweet_features = ret_nodes['tweet_features']
    hashtag_features = ret_nodes['hashtag_features']
    
    user_features=torch.tensor(user_features)
    tweet_features=torch.tensor(tweet_features)
    hashtag_features=torch.tensor(hashtag_features)

    data['user'].num_nodes =user_features
    data['tweet'].num_nodes=tweet_features
    data['hashtag_features'].num_nodes=hashtag_features

    data['user','otweets','tweet'].edge_index=edge_user_otweet
    data['user','otweets','tweet'].edge_label=label_user_otweet

    data['user','qtweets','tweet'].edge_index=edge_user_qtweet
    data['user','qtweets','tweet'].edge_label=label_user_qtweet

    data['user','ctweets','tweet'].edge_index=edge_user_ctweet
    data['user','ctweets','tweet'].edge_label=label_user_ctweet

    data['user','rtweets','tweet'].edge_index=edge_user_rtweet
    data['user','rtweets','tweet'].edge_label=label_user_rtweet

    data['tweet','quotes','tweet'].edge_index=edge_qtweet_tweet
    data['tweet','quotes','tweet'].edge_label=label_qtweet_tweet

    data['tweet','comments','tweet'].edge_index=edge_ctweet_tweet
    data['tweet','comments','tweet'].edge_label=label_ctweet_tweet

    data['tweet','mentions','user'].edge_index=edge_tweet_user
    data['tweet','mentions','user'].edge_label=label_tweet_user

    data['hashtag','hashtagedge','tweet'].edge_index=edge_hashtag_tweet
    data['hashtag','hashtagedge','tweet'].edge_label=label_hashtag_tweet

    with open('graph.pickle', 'wb') as fp:
        pickle.dump(data, fp)
