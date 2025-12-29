#!/usr/bin/env python
# coding: utf-8

import ast
from pathlib import Path
import networkx as nx
import csv
import numpy as np
import pandas as pd
import os
import glob
import csv
from pathlib import Path
from graphviz import Source
import networkx as nx
# import matplotlib.pyplot as plt
# from pyvis.network import Network
import re
from pandas import ExcelWriter
import xlsxwriter
# import tweepy
import json
import pandas as pd
import csv
import re
import string
import os
import os.path
# import time
from time import strftime, localtime
from datetime import datetime, timezone


# save graph for reference
# nx.write_gexf(G, "/home/protests/Documents/hashtag_scrape/all_rts_ids_new.gexf")
# nx.write_graphml_lxml(G,  "/home/protests/Documents/hashtag_scrape/all_rts_ids_new.graphml")
edge_dt = pd.DataFrame(columns=['user', 'mentioned', 'weight'])
node_dt = pd.DataFrame(columns=['data', 'info'])

G = nx.read_gexf("/home/dipanjan/Network_Analysis/all_rts_ids_new.gexf")

#av = nx.average_clustering(G, weight='weight', count_zeros=True)
#ith_info = ['av_clustering (count_zeros=True)', av]
#node_dt.loc[len(node_dt)] = ith_info

# av2 = nx.average_clustering(G, weight='weight', count_zeros=False)
# ith_info = ['av_clustering (count_zeros=False)', av2]
# node_dt.loc[len(node_dt)] = ith_info
G = nx.DiGraph(G)
print("Read Graph")

d = nx.density(G)
ith_info = ['graph density', d]
node_dt.loc[len(node_dt)] = ith_info

info = nx.info(G)
ith_info = ['graph info', info]
node_dt.loc[len(node_dt)] = ith_info
print("got density")
dfObj = pd.DataFrame()

li = list(G.nodes())

in_deg = []
dic = dict(G.in_degree(weight='weight'))
for key, value in dic.items():
    in_deg.append(value)

print("Got Indegree")
out_deg = []
dic = dict(G.out_degree(weight='weight'))
for key, value in dic.items():
    out_deg.append(value)
print("Got outdegree")


centrality = []
d = dict(nx.eigenvector_centrality(G, weight='weight', max_iter=600))
for key, value in d.items():
    centrality.append(value)

print("Got centtrality")

#cluster_coeff = []
#d = dict(nx.clustering(G,  weight='weight'))
#for key, value in d.items():
 #   cluster_coeff.append(value)



bt_centrality = []
d = dict(nx.betweenness_centrality(G,weight='weight'))
for key, value in d.items():
    bt_centrality.append(value)
print("got bt centrality")

# pr = []
# d = dict(nx.pagerank(G, weight='weight', max_iter=600)) #0.85 default
# for key, value in d.items():
#     pr.append(value)

# h, a = nx.hits(G, max_iter=1000)
# hubs = []
# auths = []

# d = dict(h)
# for key, value in d.items():
#     hubs.append(value)
# d = dict(a)
# for key, value in d.items():
#     auths.append(value)

dfObj['node'] = li
dfObj['in degree'] = in_deg
dfObj['out degree'] = out_deg
dfObj['centrality'] = centrality
#dfObj['cluster_coeff'] = cluster_coeff
dfObj['bt_centrality'] = bt_centrality
# dfObj['page rank'] = pr
# dfObj['hub'] = hubs
# dfObj['authority'] = auths


edges = G.edges()
for u, v in edges:
    ith_tweet = [u, v, G[u][v]['weight']]
    edge_dt.loc[len(edge_dt)] = ith_tweet

dfObj.to_csv("/home/dipanjan/Network_Analysis/all_rts_ids_new.csv",encoding="utf-8-sig", index=False)

writer = pd.ExcelWriter("/home/dipanjan/Network_Analysis/all_rts_ids_new.xlsx", engine='xlsxwriter')
# dfObj.to_excel(writer, sheet_name="node_data", encoding="utf-8-sig", index=False)

edge_dt.to_excel(writer, sheet_name="edge_data", encoding="utf-8-sig", index=False)
# node_dt.to_excel(writer, sheet_name="graph_data", encoding="utf-8-sig", index=False)
writer.save()

print("SAVED: /home/protests/Documents/hashtag_scrape/all_rts.xlsx")




