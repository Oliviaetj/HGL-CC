import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
G = nx.MultiDiGraph()
#中间过程
row=np.array(['CO_sensor','thermometer','fire_alarm','attacker','humidity_sensor','ventilator','heater','air_conditioner','occupancy_sensor','entrance_guard','light0','light1','brightness_sensor','light_controller'])
G.add_nodes_from(['CO_sensor','thermometer','fire_alarm','attacker','humidity_sensor','ventilator','heater','air_conditioner','occupancy_sensor','entrance_guard','light0','light1','brightness_sensor','light_controller'])
value=np.random.randn(196)
for j in range(0,14):
    for i in range(0,14):
        if i!=j:
            G.add_weighted_edges_from([(row[j],row[i],value[14*j+i])])#边的起点，终点，权重
        else:
            pass
node_sizes = [44,22,29,20,35,11,32,20,41,22,57,10,11,20]
node_sizesnew=[]
print(type(G))
for i in node_sizes:
    i=i*20
    node_sizesnew.append(i)
pos={'CO_sensor':(37.29,77.42),'thermometer':(31.73,59.91),'fire_alarm':(40.47,27.53),'attacker':(58.82,15.55),'humidity_sensor':(46.29,89.65),'ventilator':(72,52),'heater':(61.64,45.73),'air_conditioner':(59.22,59.78),'occupancy_sensor':(11.64,49.73),'entrance_guard':(50.55,49.27),'light0':(46.56,56.18),'light1':(74.75,74.25),'brightness_sensor':(58.25,46),'light_controller':(49.09,61.09)}
nx.draw_networkx_nodes(G,pos,node_size=node_sizesnew,alpha=0.4)
nx.draw_networkx_labels(G,pos,font_size=8,)
nx.draw(G,pos,edge_color='lightseagreen',alpha=0.5,connectionstyle='arc3, rad = 0.2',width=[float(v['weight']) for (r,c,v) in G.edges(data=True)])
edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])
plt.savefig("D:/MATCH52.png",dpi=200, bbox_inches='tight')
plt.show()
print('finish')

nx.draw(G,pos,connectionstyle='arc3, rad = 0.2',width=[float(v['weight']) for (r,c,v) in G.edges(data=True)])
#connectionstyle='arc3, rad = 0.2'arc控制双向，rad调线条弧度
