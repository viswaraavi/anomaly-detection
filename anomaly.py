import hashlib
import networkx as nx
import igraph
import glob
import numpy as np
import sys
from matplotlib import pyplot as plt

#This is the code for reading the graph
def graph_read(fname):

    graph = nx.DiGraph()
    f = open(fname,'r')
    for line in f:
        e1,e2 = [int(x) for x in line.split(' ')]
        graph.add_edge(e1,e2)
    f.close()
    return graph

#This function generates the feature set that is needed for generating simhash

def feature_set(graph):  
    pageranks =  nx.pagerank(graph) 
    features = [(str(ti),wi) for ti,wi in pageranks.items()]
    for E in graph.edges():
        ti = str(E[0])+str(E[1])
        outlinks = graph.out_degree(E[0])       
        wi = pageranks[E[0]]/outlinks 
        features.append((ti,wi))
    return features

#This function generates the hash value

def simhash(token):
    h_weight=[0]*64
    masks = [1 << i for i in range(64)]
    ve_hash=[(int(hashlib.md5(i.encode('utf-8')).hexdigest(),16),j) for i,j in token]

    for hash1,quality in ve_hash:
        for j in range(64):
            h_weight[j]+=quality if hash1 & masks[j] else -quality

    b_weight=0
    for i in range(64):
        if h_weight[i]>=0:
            b_weight |=masks[i]
    return b_weight

#This calculates the simmilarity measure                
def hamdist(list1, list2):

    x = (list1 ^ list2) & ((1 << 64) - 1)
    ans = 0.0
    while x:
        ans += 1
        x &= x - 1
    return 1-(ans/64)

# This function calculates the anomolous points
def anomalous_graph(compare, lw_thresh,up_thresh):
    anomalous_graph = []
    for index, similarity in enumerate( compare[:-1] ): 
        similarity1 = compare[index+1]
        if ((similarity < lw_thresh) or (similarity > up_thresh)) and ((similarity1 < lw_thresh) or (similarity1 > up_thresh)):
            anomalous_graph.append(index+1 )
    return anomalous_graph

#The main program

list_of_files=glob.glob("/home/viswa/Desktop/datasets/datasets/"+sys.argv[1]+"/*.txt")
list_of_files=sorted(list_of_files, key=lambda x: int(x.split('/')[7].split('_')[0]))

#The compare contains the list of the simmilarity values
compare=[]
for i,j in zip(list_of_files,list_of_files[1:]):
    graph1=graph_read(i)
    graph2=graph_read(j)
    features1=feature_set(graph1)
    features2=feature_set(graph2)
    k=hamdist(simhash(features1),simhash(features2))
    compare.append(k)



#Calculating the moving average
sum1=0.0
for i,j in zip(compare,compare[1:]):
    sum1=sum1+abs((j-i))
    

#Calculating the lower threshold and the upper threshold
M=sum1/(len(compare)-1)
lower_thresh=np.median(compare)-2*M
upper_thresh=np.median(compare)+2*M

#Printing the output

print "Now printing the value of M moving average "
print M
print "Now printing the lower threshold value"
print lower_thresh
print "Now printing the upper threshold value"
print upper_thresh
print "Now printing the list of indices of anomolous graphs"
print anomalous_graph(compare,lower_thresh,upper_thresh)

outfile = open(sys.argv[1]+"_time_series.txt" , 'w+') 
outfile.write(str(compare))


x=range(0,len(compare))
plt.plot(x,compare)
plt.plot(x,[lower_thresh.tolist()]*len(x),color='red')
plt.plot(x,[upper_thresh.tolist()]*len(x),color='red')
plt.show()


