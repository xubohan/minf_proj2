import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
import json





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


eis =  np.array([2,4,5,11,6,7,8,3,0,9],dtype=np.float32)
# eis = np.full(shape=N, fill_value=1.0, dtype=np.float32)
# eis = np.tile(eis, N)
eid =  np.array([0,1,1, 1,1,1,2,2,3,3],dtype=np.float32)
# eid = np.full(shape=N, fill_value=1.0, dtype=np.float32)
# eid = np.tile(eid, N)


num_per_layer = np.full(shape=2, fill_value=1.0, dtype=np.float32)
iseed = np.array([1], dtype=np.int32)


runner = SdkRuntime(args.name, cmaddr = args.cmaddr)

#get symbols on device
x_symbol = runner.get_id('x')
eis_symbol = runner.get_id('eis')
eid_symbol = runner.get_id('eid')
npl_symbol = runner.get_id('npl')
is_symbol = runner.get_id('iseed')


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

total_sampled = 3

seis_res = np.zeros([total_sampled*width], dtype=np.float32)
seid_res = np.zeros([total_sampled*width], dtype=np.float32)
ssize_res = np.zeros([2*width], dtype=np.float32)
snid_res = np.zeros([total_sampled * 2*width], dtype=np.float32)

sndseis_res = np.zeros([total_sampled*total_sampled*width], dtype=np.float32)
sndseid_res = np.zeros([total_sampled*total_sampled*width], dtype=np.float32)
sndssize_res = np.zeros([2*width], dtype=np.float32)
sndsnid_res = np.zeros([total_sampled*total_sampled* 2*width], dtype=np.float32)


runner.load()
runner.run()

x_result = np.zeros([1], dtype=np.float32)
eis_res = np.zeros([10], dtype=np.float32)


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

#This is for multiple PEs
runner.memcpy_h2d(eis_symbol, np.tile(eis,width), 0, 0, width, 1, 10, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


runner.memcpy_h2d(eid_symbol, np.tile(eid,width), 0, 0, width, 1, 10, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  
runner.memcpy_h2d(npl_symbol, np.tile(num_per_layer,width), 0, 0, width, 1, 2, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_h2d(is_symbol, np.tile(iseed,width), 0, 0, width, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

#launch the function

runner.launch('compute', nonblock=False)


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


runner.memcpy_d2h(seis_res, sampled_eis_symbol, 0, 0, width, 1, total_sampled, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(seid_res, sampled_eid_symbol, 0, 0, width, 1, total_sampled, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(snid_res, sampled_nid_symbol, 0, 0, width, 1, total_sampled*2, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(ssize_res, sampled_size_symbol, 0, 0, width, 1, 2, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(sndseis_res, snd_sampled_eis_symbol, 0, 0, width, 1, total_sampled*total_sampled, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(sndseid_res, snd_sampled_eid_symbol, 0, 0, width, 1, total_sampled*total_sampled, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(sndsnid_res, snd_sampled_nid_symbol, 0, 0, width, 1, total_sampled*total_sampled*2, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.memcpy_d2h(sndssize_res, snd_sampled_size_symbol, 0, 0, width, 1, 2, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  
runner.stop()

#print(eis_res)
# print('src:',seis_res)
# print('dst:',seid_res)
# print('size:', ssize_res)
# print('snid:', snid_res)

# print('org_src:',snid_res[0],snid_res[1])

# print("Second sampling")
# print('src:',sndseis_res)
# print('dst:',sndseid_res)
# print('size:', sndssize_res)
# print('snid:', sndsnid_res)


































