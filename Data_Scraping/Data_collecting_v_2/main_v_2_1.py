## This is a version for layer 2 since there are too many videos to scrape, we need to control the speed to avoid anti-scrape
import requests
import os
import time
import pandas as pd
import csv
import numpy as np
import sys

api = 'https://api.bilibili.com/x/web-interface/archive/related?'
api2 = 'https://api.bilibili.com/x/web-interface/view?bvid='
keys = ['node','aid','title','ctime','tid','tname','pubdate','duration']
current_path = os.path.dirname(__file__)

start = 19889

file_layer_p = current_path+'/layer/%s_layer.csv' %(1)
previous_layer = pd.read_csv(file_layer_p)

for i in [0,1,2,3,4]:
    
    if i != 4:

        IDs = previous_layer[start+i*10000:start+10000]['aid'].values.tolist()
        
    else:
        
        IDs = previous_layer[start+i*10000:start+40000+2216]['aid'].values.tolist()
    
    n=1
    
    file_layer = current_path+'/layer/%s_layer_%s_%s.csv' %(2,start+i*10000,start+(i+1)*10000)
    
    while n <= 10000:
        
        n = 1

        for ID in IDs:

            url = api + "aid=" + str(ID)
                
            r = requests.get(url)
            dic = r.json()
            n += 1
            
            if dic['code'] == 0:
            
                if os.path.exists(file_layer):
                    
                    with open(file_layer, 'a', encoding = "utf-8-sig", newline="") as myFile:
                        
                        writer = csv.writer(myFile)

                        for data in dic['data']:

                                
                            data['node'] = str(ID)
                            
                            writer.writerow([data.get(key,np.nan) for key in keys])
                else:
            
                    with open(file_layer, 'a', encoding = "utf-8-sig", newline="") as myFile:
                
                        writer = csv.writer(myFile)
                
                        writer.writerow(keys)
                        
                        for data in dic['data']:
                            
                               
                            data['node'] = str(ID)
                            
                            writer.writerow([data.get(key,np.nan) for key in keys])
            
            else:
                
                print("error")
                
                print(ID)
                
                print(n)
                
                requests.post('https://api.day.app/rL9FQVe3nAASeaxics3xHe/Error/Error in the second layer get!Layer:%s.Number:%s.AID:%s' %(i,n,ID))
                        
                time.sleep(1)

                sys.exit
                
                
            if n % 5000 == 0:
                
                requests.post('https://api.day.app/rL9FQVe3nAASeaxics3xHe/Success/FINISHED ID: %s to %s' %(start+i*10000+n-5000,start+i*10000+n))
                
                time.sleep(2400)
            
            elif n % 2000 == 0:
                
                time.sleep(1200)    
                            
            elif n % 400 == 0:
                
                time.sleep(120)
            
            elif n % 100 == 0:
                
                time.sleep(60)
                
            elif n % 10 == 0:
                
                time.sleep(10)
                
            elif n % 1 == 0:
                
                time.sleep(0.5)
            
        time.sleep(1800)