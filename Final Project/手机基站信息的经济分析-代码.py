import pandas as pd
from pyecharts import options as opts
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode
from pyecharts.globals import SymbolType
from pyecharts.globals import ThemeType
import osmnx as ox
from pyecharts.charts import BMap
from pyecharts import options as opts
from pyecharts.charts import Page, ThemeRiver
from pyecharts.globals import ChartType, SymbolType, GeoType
from pyecharts.globals import GeoType      
import os 
import sys
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def locate():
    # 对数据进行绘图
    station_data = pd.read_csv("station.csv",header = 0, sep = '\t')
    latitude = station_data['lat'].values
    longitude = station_data['lng'].values
    index = station_data['station_id'].values
    length = len(latitude)
    mag = [1] * length
    #可视化
    BMaps = BMap()
    data_pair = dict.fromkeys(index, 0)

    user = pd.read_csv("mobile.csv",header = 0, sep = '\t')
    locate = user['station_id'].values
    count = user['count'].values

    for i in range(len(locate)):
        try:
            if count[i] >= 10:
                data_pair[locate[i]] += int(count[i])
        except:
            pass

    for i in range(len(index)):
        BMaps.add_coordinate(index[i], longitude[i], latitude[i])

    BMaps.add_schema(baidu_ak = 'UoW1wfwAiRoBPYeh52XzdUGGoByxmUN2',
                    center=(121.37, 31.23),
                    zoom=10,
                    map_style={
            "styleJson": [
                {
                    "featureType": "water",
                    "elementType": "all",
                    "stylers": {"color": "#031628"},
                },
                {
                    "featureType": "land",
                    "elementType": "geometry",
                    "stylers": {"color": "#000102"},
                },
                {
                    "featureType": "highway",
                    "elementType": "all",
                    "stylers": {"color": "#50A3BA"},
                },

                {
                    "featureType": "local",
                    "elementType": "geometry",
                    "stylers": {"color": "#000000"},
                },
                {
                    "featureType": "railway",
                    "elementType": "geometry.fill",
                    "stylers": {"color": "#000000"},
                },
                {
                    "featureType": "railway",
                    "elementType": "geometry.stroke",
                    "stylers": {"color": "#08304b"},
                },
                {
                    "featureType": "building",
                    "elementType": "geometry.fill",
                    "stylers": {"color": "#000000"},
                },
                {
                    "featureType": "all",
                    "elementType": "labels.text.fill",
                    "stylers": {"color": "#857f7f"},
                },
                {
                    "featureType": "all",
                    "elementType": "labels.text.stroke",
                    "stylers": {"color": "#000000"},
                },
                {
                    "featureType": "building",
                    "elementType": "geometry",
                    "stylers": {"color": "#022338"},
                },
                {
                    "featureType": "green",
                    "elementType": "geometry",
                    "stylers": {"color": "#062032"},
                },
                {
                    "featureType": "boundary",
                    "elementType": "all",
                    "stylers": {"color": "#465b6c"},
                },
                {
                    "featureType": "manmade",
                    "elementType": "all",
                    "stylers": {"color": "#022338"},
                },
                {
                    "featureType": "label",
                    "elementType": "all",
                    "stylers": {"visibility": "on"},
                },
            ]
        })

    # 添加数据点
    data_pair = [list(z) for z in zip(data_pair.keys(), data_pair.values())]
    BMaps.add('次数', data_pair,type_="scatter",symbol_size=3, label_opts=opts.LabelOpts(is_show=False))
    
    pieces = [
            {'max': 10000, 'label': '10000以下', 'color': '#50A3BA'},
            {'min': 10000, 'max': 20000, 'label': '10000-20000', 'color': '#E2C568'},
            {'min': 20000, 'max': 30000, 'label': '20000-30000', 'color': '#D94E5D'},
            {'min': 30000, 'max': 40000, 'label': '30000-40000', 'color': '#3700A4'},
            {'min': 40000, 'label': '40000+', 'color': '#81AE9F'},
        ]

    BMaps.set_global_opts(
            visualmap_opts=opts.VisualMapOpts(is_piecewise=True, pieces= pieces, pos_top='top', pos_left = 'right'),
            title_opts=opts.TitleOpts(title="基站频率分布>10")
        )
    #导出图像
    BMaps.render('基站频率分布10.html')

def osmnx_store():
    # 提取出相应地区路网并储存
    place = ['松江区, 上海, 中国', '闵行区, 上海, 中国', '松江区, 上海, 中国', '浦东新区, 上海, 中国',\
        '虹口区, 上海, 中国', '普陀区, 上海, 中国', '宝山区, 上海, 中国','崇明区, 上海市, 中国', '徐汇区, 上海市, 中国','静安区, 上海市, 中国']
    place = ['徐汇区, 上海市, 中国','静安区, 上海市, 中国']
    for place1 in place:
        G = ox.graph_from_place(place1, network_type='drive', which_result=2)
        G_projected = ox.project_graph(G)
        G_project_file=open( place1 + '_project.pickle','wb')
        pickle.dump(G_projected,G_project_file)
        G_project_file.close()
        print(place1)


def cal_route( origin_point, destination_point, G):
    # calculate the nearest route and distance between start_point and end_point in G
    origin_node = ox.get_nearest_node(G, origin_point)
    destination_node = ox.get_nearest_node(G, destination_point)
    route = nx.shortest_path(G, origin_node, destination_node, weight='length')
    distance = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
    return route, distance

def minimun_tree():
    # 计算最小生成树
    file=open('G_project.pickle','rb')
    G=pickle.load(file)
    file.close()
    station_data = pd.read_csv("station.csv",header = 0, sep = '\t')
    latitude = station_data['lat'].values
    longitude = station_data['lng'].values
    index = station_data['station_id'].values
    with open('mobile10.txt') as f:
        lines = f.readlines()
    i = 286
    while i < len(lines):
        distance = 0 # store the distance
        total_route = []

        processed_node = []

        id_index, length = lines[i].rstrip('\n').split() 
        i += 1
        length = int(length)
        if length  == 0:
            i += 1
            continue
        node_locate = [] #store the location
        nodes = lines[i].rstrip('\n').split()
        for node in nodes:
            # turn the index into location
            node_index = np.where(index == node)
            node_locate.append((latitude[node_index][0], longitude[node_index][0]))
        processed_node.append(node_locate[0])
        
        for j in range(1, length):
            min_distance = 99999999
            min_path = []
            for node in processed_node:
                try:
                    route, distance = cal_route(node, node_locate[j], G)
                except:
                    route = []
                    min_distance = 99999999
                if distance < min_distance:
                    min_distance = distance
                    min_path = route 
            processed_node.append(node_locate[j])
            distance += min_distance
            total_route += min_path
        with open('route10.txt', 'a+') as f:
            f.write(str(id_index) + ' ' + str(distance) + '\n')
            for j in total_route:
                f.write( str(j) + ' ')
            f.write('\n')
        i += 1

def osmnx_test():
    # 进行一些关于osmnx的尝试
    '''
    G = ox.graph_from_place('Shanghai, China', which_result=2, network_type='drive')
    c
    ox.save_graphml(G, filename='STREETGRAPH_FILENAME')
    '''
    '''
    G_project_file=open('interested_project.pickle','wb')
    pickle.dump(G,G_project_file)
    G_project_file.close()
    '''
    
    #place = ['松江区, 上海, 中国', '闵行区, 上海, 中国', '浦东新区, 上海, 中国',\
    #    '虹口区, 上海, 中国', '普陀区, 上海, 中国', '宝山区, 上海, 中国','崇明区, 上海市, 中国', '徐汇区, 上海市, 中国','静安区, 上海市, 中国']


    file=open('G_project.pickle','rb')
    G=pickle.load(file)
    file.close()

    origin_point = (31.2827320285,121.2106132507)
    destination_point = (31.2819500000,121.1651300000)

    fig, ax = ox.plot_graph_route(G, route, origin_point=origin_point, destination_point=destination_point)
    '''
    # calculate basic network stats
    stats = ox.basic_stats(G)
    extend_stats = ox.extended_stats(G, ecc=True, bc=True, cc=True)
    '''
    G_projected = ox.project_graph(G)
    edge_centrality = nx.edge_betweenness_centrality(G_projected)

    for key, value in extend_stats.items():
        stats[key] = value
    pd.Series(stats)

def filtering():
    # 只留下计数值大于10的前五个基站位置点
    user = pd.read_csv("mobile.csv",header = 0, sep = '\t')
    locate = user['station_id'].values
    count = user['count'].values
    user_id = user['user_id'].values
    station_data = pd.read_csv("station.csv",header = 0, sep = '\t')
    latitude = station_data['lat'].values
    longitude = station_data['lng'].values
    index = station_data['station_id'].values
    
    temper = 1
    length = 0
    store_index = []
    for i in range(len(user_id)):
        if int(user_id[i]) != temper:
            # write
            with open('mobile10.txt', 'a+') as f:
                f.write(str(temper) + ' ' + str(length) + '\n')
                for j in store_index:
                    f.write( j + ' ')
                f.write('\n')
            # update
            temper = int(user_id[i])
            length = 0
            store_index = []
            if (count[i] >= 10) and (locate[i] in index):
                length += 1
                store_index.append(locate[i])
        else:
            if count[i] >= 10 and (locate[i] in index):
                length += 1
                store_index.append(locate[i])

def see():
    #可视化综合
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    file=open('route.pickle','rb')
    route = pickle.load(file)
    file.close()
    file=open('TOTAL.pickle','rb')
    TOTAL = pickle.load(file)
    file.close()
    file=open('G_project.pickle','rb')
    G = pickle.load(file)
    file.close()
    
    values = []
    '''
    for key, value in route.items():
        if True or (key in TOTAL.keys()):
            values.append(int(value))
    '''
    sorted_index = sorted(TOTAL.items(), key=lambda item:item[1], reverse=True)[0:500]
    #index = filter(lambda x:1159== x[1], route.items())
    write_index = []
    for i,j in sorted_index:
        write_index.append(i)

    nc = ['#42CD6D' if node in write_index else '#336699' for node in G.nodes()]
    ns = [50 if node in write_index else 6 for node in G.nodes()]
    fig, ax = ox.plot_graph(G, node_size = ns, node_color=nc, node_zorder=2)
    '''
    '''
    plt.hist(values, bins='auto', facecolor="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel('途经次数')
    plt.ylabel('频数')
    plt.title('次数分布直方图')
    plt.show()
    fig, ax = ox.plot_graph(ox.project_graph(G))
    '''
    write_index = sorted(list(TOTAL.values()), reverse =True)
    x = range(1, len(write_index) + 1)
    plt.title(u'综合道路中心性指标分布')
    plt.xlabel('指标排序')
    plt.ylabel('中心性指标值')
    # plt.scatter(x, y, s, c, marker)
    # x: x轴坐标
    # y：y轴坐标
    # s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
    # c: 点的颜色 
    # marker: 标记的样式 默认是 'o'
    plt.legend()

    plt.scatter(x, write_index, s=2, c="black", marker='o')
    plt.show()
    '''

def corre_cal():
    # 计算相关系数
    file=open('TOTAL.pickle','rb')
    TOTAL = pickle.load(file)
    file.close()
    file=open('BC.pickle','rb')
    BC = pickle.load(file)
    file.close()
    file=open('CC.pickle','rb')
    CC = pickle.load(file)
    file.close()
    file=open('Pagerank.pickle','rb')
    pagerank = pickle.load(file)
    file.close()

    file=open('route.pickle','rb')
    route = pickle.load(file)
    file.close()

    route_value = []
    pr = []
    bc = []
    cc = []
    total = []
    for i in BC.keys():
        try:
            route_value.append(route[i])
            pr.append(pagerank[i])
            bc.append(BC[i])
            cc.append(CC[i])
            total.append(TOTAL[i])
        except:
            pass
    route_pd = pd.Series(route_value)
    pr_pd = pd.Series(pr)
    bc_pd = pd.Series(bc)
    cc_pd = pd.Series(cc)
    TOTAL = pd.Series(total)    
    aa = 1

def comparation():
    # 进行道路等级比较并绘图
    file=open('TOTAL.pickle','rb')
    TOTAL = pickle.load(file)
    file.close()
    file=open('route.pickle','rb')
    route = pickle.load(file)
    file.close()
    file=open('G_project.pickle','rb')
    G = pickle.load(file)
    file.close()
    route_value = []
    total_value = []
    route_sorted = sorted(route.items(), key=lambda item:item[1], reverse=True)
    total_sorted = sorted(TOTAL.items(), key=lambda item:item[1], reverse=True)
    for key,value in route_sorted:
        route_value.append(key)
    for key,value in total_sorted:
        total_value.append(key)

    route_value_high = route_value[0:6400]
    total_value_high = total_value[0:3700]
    route_value_low = route_value[-6400:-1]
    total_value_low = total_value[-3700:-1]
    high_high = list(set(total_value_high).intersection(set(route_value_high)))
    high_low = list(set(total_value_high).intersection(set(route_value_low)))
    low_high = list(set(total_value_low).intersection(set(route_value_high)))
    low_low = list(set(total_value_low).intersection(set(route_value_low)))
    # 在路线图上可视化标记点
    nc = ['#42CD6D' if node in low_low else '#336699' for node in G.nodes()]
    ns = [50 if node in low_low else 6 for node in G.nodes()]
    fig, ax = ox.plot_graph(G, node_size = ns, node_color=nc, node_zorder=2)

    aa= 1

def given_weight():
    #为中心性指标赋权
    Pagerank = {}
    BC = {}
    CC = {}
    total = {}
    places = ['虹口区, 上海, 中国', '普陀区, 上海, 中国', '宝山区, 上海, 中国','崇明区, 上海市, 中国', \
        '徐汇区, 上海市, 中国','静安区, 上海市, 中国', '闵行区, 上海, 中国', '松江区, 上海, 中国']
    street = 0
    for place in places:
        file=open(place + '_information.pickle','rb')
        G=pickle.load(file)
        file.close()
        street += sum(G['streets_per_node_counts'].values())
        print(place + str(sum(G['streets_per_node_counts'].values())))
        for key, value in G['pagerank'].items():
            Pagerank[key] = value
        for key, value in G['closeness_centrality'].items():
            CC[key] = value
        for key, value in G['betweenness_centrality'].items():
            BC[key] = value
        
        pr = list(G['pagerank'].values())
        bc = list(G['betweenness_centrality'].values())
        cc = list(G['closeness_centrality'].values())
        pr_pd = pd.Series(pr)
        bc_pd = pd.Series(bc)
        cc_pd = pd.Series(cc)
        pr_std = np.std(pr) 
        bc_std = np.std(bc)
        cc_std = np.std(cc)
        # 计算相关系数
        pr_weight = pr_std * (1-pr_pd.corr(bc_pd)) * (1-pr_pd.corr(cc_pd))
        bc_weight = bc_std * (1-bc_pd.corr(pr_pd)) * (1-bc_pd.corr(cc_pd))
        cc_weight = cc_std * (1-cc_pd.corr(pr_pd)) * (1-cc_pd.corr(bc_pd))
        total_sum = pr_weight + bc_weight + cc_weight
        pr_weight /= total_sum
        bc_weight /= total_sum
        cc_weight /= total_sum

        for key in G['pagerank'].keys():
            total[key] = G['pagerank'][key] * pr_weight + G['betweenness_centrality'][key] * bc_weight + G['closeness_centrality'][key] * cc_weight
        print(place, pr_weight, bc_weight, cc_weight)
    '''
    Pagerank_file=open('Pagerank.pickle','wb')
    pickle.dump(Pagerank, Pagerank_file)
    Pagerank_file.close()

    BC_file=open('BC.pickle','wb')
    pickle.dump(BC, BC_file)
    Pagerank_file.close()

    CC_file=open('CC.pickle','wb')
    pickle.dump(CC, CC_file)
    CC_file.close()

    TOTAL_file=open('TOTAL.pickle','wb')
    pickle.dump(total, TOTAL_file)
    TOTAL_file.close()
    '''

if __name__ == "__main__":
    os.chdir(sys.path[0])
    #locate()
    #filtering()
    #osmnx_test()
    #osmnx_store()
    #minimun_tree()
    #see()
    #corre_cal()
    #comparation()
    #given_weight()