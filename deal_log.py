import os
import re
from matplotlib import pyplot as plt
import numpy as np

# with open("log","r",encoding="utf-8")as f:
#     contents = f.read()
# pattern = "step \d*?:loss:.*?\|"
# energys = re.findall(pattern,contents)
# energys = [float(re.sub("step \d*?:loss:","",e).replace("|","")) for e in energys]
# print(len(energys))
#
# with open("logc","r",encoding="utf-8")as f:
#     contents = f.read()
# pattern = "step \d*?:loss:.*?\|"
# energys_1 = re.findall(pattern,contents)
# energys_1 = [float(re.sub("step \d*?:loss:","",e).replace("|","")) for e in energys_1]
# print(len(energys_1))
#
# plt.figure()
# plt.plot(np.array(energys), c='r', label='rl loss in mnist')
# plt.plot(np.array(energys_1), c='b', label='rl loss in cifar10')
# plt.legend(loc='best')
# plt.ylabel('loss_value')
# plt.xlabel('training step')
# plt.grid()
# plt.savefig(f"rl_loss.png")
#
# with open("log","r",encoding="utf-8")as f:
#     contents = f.read()
# pattern = "total cost energy:.*? "
# energys = re.findall(pattern,contents)
# energys = [float(e.replace('total cost energy:','').replace(' ','')) for e in energys]
# total_energys_1 = []
# for i in range(0,len(energys),60):
#     ep_e = 0
#     for j in range(60):
#         ep_e+=energys[i+j]
#     print(ep_e)
#     total_energys_1.append(ep_e)
#
# print(total_energys_1)
#
# with open("logc","r",encoding="utf-8")as f:
#     contents = f.read()
# pattern = "total cost energy:.*? "
# energys = re.findall(pattern,contents)
# energys = [float(e.replace('total cost energy:','').replace(' ','')) for e in energys]
# total_energys_2 = []
# for i in range(0,len(energys),60):
#     ep_e = 0
#     for j in range(60):
#         ep_e += energys[i + j]
#     print(ep_e)
#     total_energys_2.append(ep_e)
#
# print(total_energys_2)
#
# plt.figure()
# plt.plot(np.array(total_energys_1[1:]), c='r', label='cost energy in mnist')
# plt.plot(np.array(total_energys_2), c='y', label='cost energy in cifar10')
# plt.legend(loc='best')
# plt.ylabel('energy_value')
# plt.xlabel('training episode')
# plt.grid()
# plt.savefig(f"rl_E.png")


with open("log","r",encoding="utf-8")as f:
    contents = f.read()
pattern = "running q:.*?\n"
energys = re.findall(pattern,contents)
energys = [float(e.replace('running q:','').replace('\n','')) for e in energys]
total_energys_1 = []
for i in range(0,len(energys),60):
    ep_e = energys[i+59]
    print(ep_e)
    total_energys_1.append(ep_e)

print(total_energys_1)

with open("logc","r",encoding="utf-8")as f:
    contents = f.read()
pattern = "running q:.*?\n"
energys = re.findall(pattern,contents)
energys = [float(e.replace('running q:','').replace('\n','')) for e in energys]
total_energys_2 = []
for i in range(0,len(energys),60):
    ep_e = energys[i + 59]
    print(ep_e)
    total_energys_2.append(ep_e)

print(total_energys_2)

plt.figure()
plt.plot(np.array(total_energys_1), c='r', label='total q in mnist')
plt.plot(np.array(total_energys_2), c='b', label='total q in cifar10')
plt.legend(loc='best')
plt.ylabel('q_value')
plt.xlabel('training episode')
plt.grid()
plt.savefig(f"rl_q.png")

# pattern = "test-dataset: loss:.*?,acc:"
# with open("log3","r",encoding="utf-8")as f:
#     content = f.read()
# losses = re.findall(pattern,content)
# losses = [float(l.replace("test-dataset: loss:","").replace(",acc:","")) for l in losses]
# print(losses)
#
# pattern = ",acc:.*?\n"
# with open("log3","r",encoding="utf-8")as f:
#     content = f.read()
# acces = re.findall(pattern,content)
# acces = [float(l.replace("\n","").replace(",acc:","")) for l in acces]
# print(acces)

