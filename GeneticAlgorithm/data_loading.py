import json

from Models.Instance import Instance
from Models.Path import Path

INSTANCES_DATA_FILENAME = "FloData/instances.json"


def load_data(filename):
    with open(filename) as file:
        file_data = json.load(file)
    return file_data


def get_paths_list(instance):
    paths_list = []
    data_cost = load_data(instance.cost_path)
    data_flow = load_data(instance.flow_path)
    for i in range(len(data_cost)):
        paths_list.append(Path(data_cost[i]['source'], data_cost[i]['dest'],
                               data_cost[i]['cost'], data_flow[i]['amount']))
    return paths_list


def get_instances():
    instances = []
    data = load_data(INSTANCES_DATA_FILENAME)
    for entry in data:
        instances.append(Instance(entry['dimX'], entry['dimY'], entry['name'], entry['machinesNumb']
                                  , entry['flow_path'], entry['cost_path']))
    return instances
