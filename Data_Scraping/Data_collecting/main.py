import requests
import os
import time
import pandas as pd
import csv
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import gravis as gv


start = 3
tasks = 7
end = start+tasks

api = 'https://api.bilibili.com/x/web-interface/archive/related?'
api2 = 'https://api.bilibili.com/x/web-interface/view?bvid='
keys = ['node','aid','title','ctime','tid','tname','pubdate','duration']
current_path = os.path.dirname(__file__)


def get_layer_1(IDs,file_layer1,i):

    n = 0

    for ID in IDs:
        
        url = api + "aid=" + str(ID)
        r = requests.get(url)
        dic = r.json()
        n += 1
        
        if dic['code'] == 0:
        
            if os.path.exists(file_layer1):
                
                with open(file_layer1, 'a', encoding = "utf-8-sig", newline="") as myFile:
                    
                    writer = csv.writer(myFile)

                    for data in dic['data']:
                        
                        data['node'] = str(ID)
                        
                        writer.writerow([data.get(key,np.nan) for key in keys])
            else:
        
                with open(file_layer1, 'a', encoding = "utf-8-sig", newline="") as myFile:
            
                    writer = csv.writer(myFile)
            
                    writer.writerow(keys)
                    
                    for data in dic['data']:
                        
                        data['node'] = str(ID)
                        
                        writer.writerow([data.get(key,np.nan) for key in keys])
        
        else:
            
            print("error")
            
            print(ID)
            
            print(i)
            
            print(n)
            
            requests.post('https://api.day.app/rL9FQVe3nAASeaxics3xHe/Error in the first layer get! Layer:%s. Number:%s. AID:%s' %(i,n,ID))
            
            time.sleep(1)

            sys.exit
            
                        
        if n % 500 == 0:
            
            time.sleep(120)
        
        if n % 100 == 0:
            
            time.sleep(50)
            
def get_layer_2(IDs,file_layer2,i):
    
    n = 0

    for ID in IDs:
        
        url = api + "aid=" + str(ID)
        r = requests.get(url)
        dic = r.json()
        n += 1
        
        if dic['code'] == 0:
        
            if os.path.exists(file_layer2):
                
                with open(file_layer2, 'a', encoding = "utf-8-sig", newline="") as myFile:
                    
                    writer = csv.writer(myFile)

                    for data in dic['data']:
                        
                        data['node'] = str(ID)
                        
                        writer.writerow([data.get(key,np.nan) for key in keys])
            else:
        
                with open(file_layer2, 'a', encoding = "utf-8-sig", newline="") as myFile:
            
                    writer = csv.writer(myFile)
            
                    writer.writerow(keys)
                    
                    for data in dic['data']:
                        
                        data['node'] = str(ID)
                        
                        writer.writerow([data.get(key,np.nan) for key in keys])
        
        else:
            
            print("error")
            
            print(ID)
            
            print(i)
            
            print(n)
            
            requests.post('https://api.day.app/rL9FQVe3nAASeaxics3xHe/Error in the second layer get!Layer:%s.Number:%s.AID:%s' %(i,n,ID))
                   
            time.sleep(1)

            sys.exit
            
                        
        if n % 404 == 0:
            
            time.sleep(120)
        
        elif n % 101 == 0:
            
            time.sleep(50)
            
        elif n % 10 == 0:
            
            time.sleep(10)
            
        elif n % 1 == 0:
            
            time.sleep(0.5)
            
def get_data(i):
    
    layer = i
    
    file_layer1 = current_path+'/layer/%s_first_layer.csv' %(layer)
    file_layer2 = current_path+'/layer/%s_second_layer.csv' %(layer)
    file_importance = current_path+'/importance/%s_importance.csv' %(layer)

    importance = pd.read_csv(file_importance)
    IDs1 = importance['aid'].tolist()
    
    get_layer_1(IDs1,file_layer1,i)
    time.sleep(60)
    
    first_layer = pd.read_csv(file_layer1)
    IDs2 = first_layer['aid'].values.tolist()
    get_layer_2(IDs2,file_layer2,i)
    
def analyze_data(i):
    
    layer = i

    file_layer1 = current_path+'/layer/%s_first_layer.csv' %(layer)
    file_layer2 = current_path+'/layer/%s_second_layer.csv' %(layer)
    file_importance = current_path+'/importance/%s_importance.csv' %(layer+1)

    fl = pd.read_csv(file_layer1)
    sl = pd.read_csv(file_layer2)

    G1 = nx.DiGraph()

    for index, row in fl.iterrows():
        aid = row['aid']
        node = row['node']
        G1.add_edge(node, aid)
        
    for index, row in sl.iterrows():
        aid = row['aid']
        node = row['node']
        G1.add_edge(node, aid) 
        
    pagerank = nx.pagerank(G1, alpha=0.85)

    pagerank_df = pd.DataFrame.from_dict(pagerank, orient='index', columns=['importance'])

    pagerank_df.index.rename('aid',inplace=True)

    top_20_nodes = pagerank_df.sort_values(by='importance', ascending=False).head(20)

    top_20_nodes.to_csv(file_importance)
    
    
    
    


for i in range(start,end+1):
    
    get_data(i)
    
    analyze_data(i)
    
    print('-------------------------------------------------')
    
    print("FINISHED LAYER: %s"%(i))
    
    print('-------------------------------------------------')
    
    time.sleep(900)