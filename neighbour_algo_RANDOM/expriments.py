import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
import json
import time
from numpy import random



##Cora test:
import os.path as osp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_cluster import random_walk

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import SAGEConv

EPS = 1e-15

path = osp.join('cora')
dataset = Planetoid(path, 'cora', transform=T.NormalizeFeatures())
data = dataset[0]


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)


train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)



# read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Matrix dimensions
# N = int(compile_data['params']['N'])
# M = int(compile_data['params']['M'])
# N=10


# Number of PEs in program
width = int(compile_data['params']['width'])

matrix = np.array(random.randint((100),size=(100,100)))

eis = np.array(matrix[0],dtype=np.float32)
# eis =  np.array([2,4,5,11,6,7,8,3,0,9],dtype=np.float32)
# eis = np.full(shape=N, fill_value=1.0, dtype=np.float32)
# eis = np.tile(eis, N)
eid = np.array(matrix[1],dtype=np.float32)
# eid =  np.array([0,1,1, 1,1,1,2,2,3,3],dtype=np.float32)
# eid = np.full(shape=N, fill_value=1.0, dtype=np.float32)
# eid = np.tile(eid, N)


num_per_layer = np.full(shape=2, fill_value=1.0, dtype=np.float32)
iseed = np.array([1], dtype=np.int32)


runner = SdkRuntime(args.name, cmaddr = args.cmaddr)

#get symbols on device
# x_symbol = runner.get_id('x')
eis_symbol = runner.get_id('eis')
eid_symbol = runner.get_id('eid')
# npl_symbol = runner.get_id('npl')
# is_symbol = runner.get_id('iseed')


#these varaibles is used for
#copying data from device to host
sampled_eis_symbol = runner.get_id('seis')
sampled_eid_symbol = runner.get_id('seid')
sampled_size_symbol = runner.get_id('ssize')
sampled_nid_symbol = runner.get_id('snid')


snd_sampled_eis_symbol = runner.get_id('sndseis')
snd_sampled_eid_symbol = runner.get_id('sndseid')
snd_sampled_size_symbol = runner.get_id('sndssize')
snd_sampled_nid_symbol = runner.get_id('sndsnid')

total_sampled = 10
data_length = 100

seis_res = np.zeros([total_sampled*width], dtype=np.float32)
seid_res = np.zeros([total_sampled*width], dtype=np.float32)
ssize_res = np.zeros([2*width], dtype=np.float32)
snid_res = np.zeros([total_sampled * 2*width], dtype=np.float32)

sndseis_res = np.zeros([total_sampled*total_sampled*width], dtype=np.float32)
sndseid_res = np.zeros([total_sampled*total_sampled*width], dtype=np.float32)
sndssize_res = np.zeros([2*width], dtype=np.float32)
sndsnid_res = np.zeros([total_sampled*total_sampled* 2*width], dtype=np.float32)

start_time = time.time()
runner.load()
runner.run()
print("run simulator time: ", round(time.time()-start_time,5))

# x_result = np.zeros([1], dtype=np.float32)
# eis_res = np.zeros([data_length], dtype=np.float32)


#copy value from host to device

# this is for single PE
# runner.memcpy_h2d(eis_symbol, eis, 0, 0, 1, 1, 10, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


# runner.memcpy_h2d(eid_symbol, eid, 0, 0, 1, 1, 10, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  
# runner.memcpy_h2d(npl_symbol, num_per_layer, 0, 0, 1, 1, 2, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# runner.memcpy_h2d(is_symbol, iseed, 0, 0, 1, 1, 1, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

start_time = time.time()
#This is for multiple PEs
runner.memcpy_h2d(eis_symbol, np.tile(eis,width), 0, 0, width, 1, data_length, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


runner.memcpy_h2d(eid_symbol, np.tile(eid,width), 0, 0, width, 1, data_length, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

print("Host to Device time: ", round(time.time()-start_time,5))
  
# runner.memcpy_h2d(npl_symbol, np.tile(num_per_layer,width), 0, 0, width, 1, 2, streaming=False,
  # order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# runner.memcpy_h2d(is_symbol, np.tile(iseed,width), 0, 0, width, 1, 1, streaming=False,
  # order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

#launch the function
for i in range(10):
    start_time = time.time()
    runner.launch('compute', nonblock=False)
    print(i, " Compute time: ", round(time.time()-start_time,5))

    start_time = time.time()
    train_loader.sample(range(1))
    print(i, ' Python Sample Time ', round(time.time()-start_time,5))

#copy value from device to host

# runner.memcpy_d2h(eis_res, eis_symbol, 0, 0, 1, 1, N, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# runner.memcpy_d2h(x_result, x_symbol, 0, 0, 1, 1, 1, streaming=False,
  # order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# print(x_result)

# iseed = np.array([7], dtype=np.int32)
# runner.memcpy_h2d(is_symbol, iseed, 0, 0, 1, 1, 1, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# #launch the function

# runner.launch('compute', nonblock=False)
# runner.memcpy_d2h(x_result, x_symbol, 0, 0, 1, 1, 1, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


# # transmit the data
# start_time = time.time()
# # print('p1')
# runner.memcpy_d2h(seis_res, sampled_eis_symbol, 0, 0, width, 1, total_sampled, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# # print('src:',seis_res)
# # print('p2')
# runner.memcpy_d2h(seid_res, sampled_eid_symbol, 0, 0, width, 1, total_sampled, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# # print('src:',seid_res)
# # print('p3')
# runner.memcpy_d2h(snid_res, sampled_nid_symbol, 0, 0, width, 1, total_sampled*2, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# # print('p4')
# runner.memcpy_d2h(ssize_res, sampled_size_symbol, 0, 0, width, 1, 2, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# # print('p5')
# runner.memcpy_d2h(sndseis_res, snd_sampled_eis_symbol, 0, 0, width, 1, total_sampled*total_sampled, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# # print('p6')/
# runner.memcpy_d2h(sndseid_res, snd_sampled_eid_symbol, 0, 0, width, 1, total_sampled*total_sampled, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# # print('p7')
# runner.memcpy_d2h(sndsnid_res, snd_sampled_nid_symbol, 0, 0, width, 1, total_sampled*total_sampled*2, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# # print('p8')
# runner.memcpy_d2h(sndssize_res, snd_sampled_size_symbol, 0, 0, width, 1, 2, streaming=False,
#   order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# print("Copy Deivce to Host time: ", round(time.time()-start_time,5))

runner.stop()

# print(eis_res)
# print('src:',seis_res)
# print('dst:',seid_res)
# print('size:', ssize_res)
# print('snid:', snid_res)

# # print('org_src:',snid_res[0],snid_res[1])

# print("Second sampling")
# print('src:',sndseis_res)
# print('dst:',sndseid_res)
# print('size:', sndssize_res)
# print('snid:', sndsnid_res)





































