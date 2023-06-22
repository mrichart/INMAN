import sys
import numpy as np

sys.path.append('../util')
#from DDPG_con import DeepDeterministicPolicyGradient
from DDPG_con_gauss import DeepDeterministicPolicyGradient

sys.path.append('../env')
from c_srv_1_reqsize_cont_act_cont import DataCenter

env = DataCenter()

discount_rate = 0.95
lr_c          = 0.001
lr_a          = 0.001
n_steps       = 60000
to_load_model = False
keep_learning = True

#select [0] for soft, [1] for hard:
replacement = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=env.one_cycle, rep_iter_c=env.one_cycle)][0]

epsilon_init  = env.action_bound / 1.0
epsilon       = epsilon_init
epsilon_diminish_step = epsilon_init / (env.one_cycle * 2.0)
epsilon_reset_step = int(3.2 * (env.one_cycle * 2.0))

def update_epsilon(epsilon, step):
    if epsilon > 0.01:
        epsilon -= epsilon_diminish_step
    if epsilon < 0.01 and step % epsilon_reset_step == 0:
        print('epsilon reset at step: ', step)
        epsilon = epsilon_init
    return epsilon

DDPG = DeepDeterministicPolicyGradient(
                n_features=env.n_features,
                n_actions=env.n_actions,
                action_bound=env.action_bound,
                replace=replacement,
                lr_c=lr_c,
                lr_a=lr_a,
                reward_decay=discount_rate,
                memory_size=2**14,
                batch_size=2**7,
                to_load_model=to_load_model,
                name_model='c_srv_1_reqsize_cont_act_cont_DDPG_gauss')

steps_to_start_learning = DDPG.batch_size * 10

s = env.reset()
r_avg = 0

for step in range(n_steps):

    a = np.clip(np.random.normal(DDPG.choose_action(s), epsilon), -env.action_bound, env.action_bound)

    epsilon = update_epsilon(epsilon, step)

    s_, r = env.step(a)
    r_avg = r_avg * 0.999 + r * 0.001
    DDPG.memory.store_transition(s, a, r, s_)
    if step > steps_to_start_learning and keep_learning:
        DDPG.learn()
    if step % env.one_cycle == 0:
        print('step: ', step,
              ', r_avg: %.4f' % r_avg)
    if r_avg > 0.995:
        break
    s = s_

DDPG.save_model()

print('r_avg: %.4f' % r_avg)

DDPG.plot_data(env.hist[:, 0:4], "Assig_vs_Req", do_not_save=True,
               labels = ['REQ1','REQ2','ASG1','ASG2'])
DDPG.plot_data(env.hist[:, 4], "SLA", do_not_save=True)
DDPG.plot_data(env.hist[:, 5], "USED", do_not_save=True)


