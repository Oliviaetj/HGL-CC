import matplotlib.pyplot as plt
import networkx as nx
import PIL


plt.rcParams['axes.unicode_minus'] = False
icons = {
'router' : 'icon/database-storage.png',
'switch': 'icon/wifi.png',
'PC': 'icon/laptop.png',
}
#载入图像
images = {k: PIL.Image.open(fname) for k, fname in icons.items()}


#创建空图
G = nx.Graph()
#创建节点
G.add_node("router" , image=images["router"])
for i in range(1,4):
    G.add_node(f"switch_{i}", image=images["switch"])
    for j in range(1,4):
        G.add_node("PC_" + str(i) + "_" + str(j), image=images["PC"])
#创建连接
G.add_edge("router" , "switch_1")
G.add_edge("router", "switch_2")
G.add_edge("router" , "switch_3")
for u in range(1,4):
    for v in range(1,4):
        G.add_edge("switch_" + str(u),"PC_" + str(u) + "_" + str(v))
# nx.draw(G, with_labels=True)

fig,ax = plt.subplots()

icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.04
icon_center = icon_size / 2.0

pos = nx.spring_layout(G, seed=1)
fig, ax = plt.subplots(figsize=(14,19))
#绘制连接
# min_source_margin和 min_target_margin调节连接端点到节点的距离
nx.draw_networkx_edges(G,pos=pos,ax=ax,arrows=True,arrowstyle="-",min_source_margin=30,min_target_margin=30,)

#给每个节点添加各自的图片
for n in G.nodes:
    xf, yf = ax.transData.transform(pos[n])#..data坐标.转..display坐标
    xa,ya = fig.transFigure.inverted().transform((xf,yf))#dis.lay坐标..转..f.igure坐标
    a = plt.axes([xa - icon_center,ya - icon_center,icon_size,icon_size])
    a.imshow(G.nodes[n]["image"])
    a.axis("off")
plt.show()
