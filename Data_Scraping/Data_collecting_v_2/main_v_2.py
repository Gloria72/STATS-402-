## This is a common version

import requests
import os
import time
import pandas as pd
import csv
import numpy as np
import sys

## Set layers and other things
start = 0
tasks = 2

end = start+tasks

api = 'https://api.bilibili.com/x/web-interface/archive/related?'
api2 = 'https://api.bilibili.com/x/web-interface/view?bvid='
keys = ['node','aid','title','ctime','tid','tname','pubdate','duration']
current_path = os.path.dirname(__file__)

## Get all videos in that layer
            
def get_layer(IDs,file_layer,i,bv=0):
    
    n = 0

    for ID in IDs:
        
        if bv==0:
            url = api + "aid=" + str(ID)
        else:
            url = api + "bvid=" + str(ID)
            
        r = requests.get(url)
        dic = r.json()
        n += 1
        
        if dic['code'] == 0:
        
            if os.path.exists(file_layer):
                
                with open(file_layer, 'a', encoding = "utf-8-sig", newline="") as myFile:
                    
                    writer = csv.writer(myFile)

                    for data in dic['data']:
                        
                        if bv == 1:
                            
                            data['node'] = requests.get(api2+ID).json()['data']['aid']
                            
                        else:
                            
                            data['node'] = str(ID)
                        
                        writer.writerow([data.get(key,np.nan) for key in keys])
            else:
        
                with open(file_layer, 'a', encoding = "utf-8-sig", newline="") as myFile:
            
                    writer = csv.writer(myFile)
            
                    writer.writerow(keys)
                    
                    for data in dic['data']:
                        
                        if bv == 1:
                            
                            data['node'] = requests.get(api2+ID).json()['data']['aid']
                            
                        else:
                            
                            data['node'] = str(ID)
                        
                        writer.writerow([data.get(key,np.nan) for key in keys])
        
        else:
            
            print("error")
            
            print(ID)
            
            print(i)
            
            print(n)
            
            requests.post('https://api.day.app/rL9FQVe3nAASeaxics3xHe/Error/Error in the second layer get!Layer:%s.Number:%s.AID:%s' %(i,n,ID))
                   
            time.sleep(1)

            sys.exit
            
            
        if n % 200001 == 0:
            
            time.sleep(1200)
        
        elif n % 2001 == 0:
            
            time.sleep(900)    
                        
        elif n % 404 == 0:
            
            time.sleep(120)
        
        elif n % 101 == 0:
            
            time.sleep(50)
            
        elif n % 10 == 0:
            
            time.sleep(10)
            
        elif n % 1 == 0:
            
            time.sleep(0.5)

## Call get layer function and input the ids in that layer

def get_data(i):
        
    layer = i
    
    file_layer = current_path+'/layer/%s_layer.csv' %(layer)
    
    if i == 0:
        
        IDs = ['BV1As4y1E71B','BV1DM4y1U7uM','BV1WV4y1Q78u','BV1WM4y1m7hj','BV18c41157x9', ## 搞笑
       'BV1Gh411G73h','BV1rV4y1f7yu','BV11j411w7a9','BV1jM411T79h','BV1Mc411s7hQ',  ## 亲子
       'BV1t84y1g7Gj','BV1qo4y1W7D1','BV1qm4y1r7ZG','BV1yh41137Zn','BV1BT411q7Zo',  ## 出行
       'BV1qT411r7LX','BV1sM411H7P4','BV1G84y1g7R5','BV1q54y1K7vQ','BV1Jv4y1L7DB',  ## 三农
       'BV1Pc411L72y','BV1Ps4y1m7jF','BV1s24y1u7Fw','BV1SY4y1S7dh','BV1zs4y1J7Uj',  ## 家居房产
       'BV1Yo4y1p7QE','BV15c41157Kf','BV1DL411o7SS','BV1XM411u7EP','BV1rv4y1G7bo',  ## 手工
       'BV1CY4y1S7PD','BV1dL411S7cK','BV1XL411Q7SS','BV1Cg4y147zr','BV1uL411r7cP',  ## 绘画
       'BV1Ga4y1T7ZC','BV1hX4y1677V','BV1F24y1w7fX','BV15M411N7MZ','BV1iM411T7sS'  ## 日常
       ]

    
        get_layer(IDs,file_layer,i,1)
        
        time.sleep(60)
        
    else:
        
        file_layer_p = current_path+'/layer/%s_layer.csv' %(layer-1)
    
        previous_layer = pd.read_csv(file_layer_p)
        IDs = previous_layer['aid'].values.tolist()
        get_layer(IDs,file_layer,i,0)
    


## Main function

for i in range(start,end+1):
    
    get_data(i)
    
    print('-------------------------------------------------')
    
    print("FINISHED LAYER: %s"%(i))
    
    requests.post('https://api.day.app/rL9FQVe3nAASeaxics3xHe/Success/FINISHED LAYER: %s' %(i))
    
    print('-------------------------------------------------')
    
    time.sleep(600)