import json
import os
from matplotlib import pyplot as plt
import numpy as np

with open("central_res.json","r",encoding="utf-8")as f:
    central_res = json.load(f)
    central_res_acc = central_res["acc record"]
    central_res_loss = central_res["loss record"]

with open("result_rl.json","r",encoding="utf-8")as f:
    result_rl = json.load(f)
    result_rl_acc = result_rl["acc"]
    result_rl_loss = result_rl["loss"]
    result_rl_e = result_rl["E_cost"]

with open("result_ea.json","r",encoding="utf-8")as f:
    result_ea = json.load(f)
    result_ea_acc = result_ea["acc"]
    result_ea_loss = result_ea["loss"]
    result_ea_e = result_ea["E_cost"]

# with open("result_random.json","r",encoding="utf-8")as f:
#     result_ra = json.load(f)
#     result_ra_acc = result_ra["acc"]
#     result_ra_loss = result_ra["loss"]
#     result_ra_e = result_ra["E_cost"]

with open("result_None.json","r",encoding="utf-8")as f:
    result_fl = json.load(f)
    result_fl_acc = result_fl["acc"]
    result_fl_loss = result_fl["loss"]
    result_fl_e = result_fl["E_cost"]

total_e = [sum(result_ea_e),sum(result_rl_e),sum(result_fl_e)]

plt.bar(["EA","BDRS","FL"], total_e, color='rbg')  # or `color=['r', 'g', 'b']`
# plt.legend(loc='best')
plt.ylabel('energy_value')
plt.xlabel('methods')
# plt.grid()
plt.savefig("FL_BDRS_ra_ea_e.png")

plt.figure()
plt.plot(np.array(result_ea_loss), c='r', label='loss of DCFL+EA')
plt.plot(np.array(result_rl_loss), c='b', label='loss of DCFL+BDRS')
# plt.plot(np.array(result_ra_loss), c='g', label='loss of DCFL+Random')
plt.plot(np.array(result_fl_loss), c='g', label='loss of FL')
plt.legend(loc='best')
plt.ylabel('loss_value')
plt.xlabel('training iteration')
plt.grid()
plt.savefig("FL_BDRS_ra_ea_loss.png")

plt.figure()
plt.plot(np.array(result_ea_acc), c='r', label='acc of DCFL+EA')
plt.plot(np.array(result_rl_acc), c='b', label='acc of DCFL+BDRS')
# plt.plot(np.array(result_ra_acc), c='g', label='acc of DCFL+Random')
plt.plot(np.array(result_fl_acc), c='g', label='acc of FL')
plt.legend(loc='best')
plt.ylabel('acc_value')
plt.xlabel('training iteration')
plt.grid()
plt.savefig("FL_BDRS_ra_ea_acc.png")

# plt.figure()
# plt.plot(np.array(result_fl_loss), c='r', label='loss of FL')
# plt.plot(np.array(result_rl_loss), c='b', label='loss of DCFL')
# plt.legend(loc='best')
# plt.ylabel('loss_value')
# plt.xlabel('training iteration')
# plt.grid()
# plt.savefig(f"FL_DCFL_loss.png")
#
plt.figure()
plt.plot(np.array(central_res_loss), c='r', label='loss of CT')
plt.plot(np.array(result_rl_loss), c='b', label='loss of DCFL')
plt.legend(loc='best')
plt.ylabel('loss_value')
plt.xlabel('training iteration')
plt.grid()
plt.savefig(f"central_DCFL_loss.png")

plt.figure()
plt.plot(np.array(central_res_acc), c='g', label='acc of CT')
plt.plot(np.array(result_rl_acc), c='y', label='acc of DCFL')
plt.legend(loc='best')
plt.ylabel('acc_value')
plt.xlabel('training iteration')
plt.grid()
plt.savefig(f"central_DCFL_acc.png")

