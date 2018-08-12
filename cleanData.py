from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

#increase recursion limit
import sys
sys.setrecursionlimit(10000)


def showCategoryBar(data):
    data_groupby = data.groupby("prime_genre")
    data_group =  data_groupby.size().reset_index(name='counts')
    pd.DataFrame(data_group.counts)

    x, y=[], []
    for index,row in data_group.iterrows():
        x.append(row['prime_genre'])
        y.append(row['counts'])
    plt.title('App Category And Accumulation')
    plt.xlabel('Category')
    plt.ylabel('Accumulation')
    plt.bar(x,y)
    plt.show()

def show_category_scatter(data):
    sns.set(style="ticks", color_codes=True)
    sns.catplot(x="prime_genre", y="rating_count_tot", data=data)
    plt.title('App Category And Accumulation')
    plt.ylabel("Rating counts (all version)")
    plt.xlabel("Category")
    plt.show()

def show_free_and_notfree(data):
    data['free'] = np.where(data['price'] == 0.00,"Yes","No")
    sns.catplot(x="free",y="rating_count_tot",order=["Yes","No"],data=data)
    plt.title('Game Rating And Price ')
    plt.ylabel("Rating counts (all version)")
    plt.xlabel("Free or Not Free")
    plt.show()


def plot3D(data,data2):
    x,y,z = data['MB'],data['price'],data['rating_count_tot']
    x2,y2,z2 = data2['MB'],data2['price'],data2['rating_count_tot']

    ax = plt.subplot(111, projection='3d')  
    
    ax.scatter(x,y,z, c='y')  
    ax.scatter(x2,y2,z2, c='r')
    plt.title('Game Rating - Price - Size ')
    ax.set_zlabel('Rating counts')  # 坐标轴
    ax.set_ylabel('Price')
    ax.set_xlabel('MB')
    plt.show()

def hierarchical_clusting(data):
    np.set_printoptions(precision=5, suppress=True)
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(data))
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

    
temp = pd.read_csv('D:\Dropbox\#NEUStuff\DataManagement\week5\AppleStore.csv')
data_frame = pd.DataFrame(temp)

#step1
# showCategoryBar(temp)

#step2
# show_category_scatter(temp)

#step3
# show_free_and_notfree(data_frame[data_frame.prime_genre == "Games"])

"""
step4 Plot the Game based on the rating
"""
# data_frame['MB'] = data_frame.size_bytes.apply(lambda x : x/(10^20))
# data_frame = data_frame[data_frame.prime_genre == 'Games']
# is_300000 = data_frame['rating_count_tot'] < 300000
# is_moretan_300000 = data_frame['rating_count_tot'] > 300000
# plot3D(data_frame[is_300000],data_frame[is_moretan_300000])

rating_language =  list(zip(data_frame['rating_count_tot'],data_frame['price']))
hierarchical_clusting(rating_language)



# Reference
#adityapatil, Visual Analysis of Apps on AppleStore, Kaggle, Retrieved from
#https://www.kaggle.com/adityapatil673/visual-analysis-of-apps-on-applestore/notebook

#Jörn,SciPy Hierarchical Clustering and Dendrogram Tutorial,Jörn's Blog ,Retrieved from
#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
