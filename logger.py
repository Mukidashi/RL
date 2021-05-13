import os
import numpy as np
import datetime,time
import matplotlib.pyplot as plt

class Logger():
    def __init__(self,save_dir):
        self.log_dir = os.path.join(save_dir,"log")
        with open(self.log_dir,"w") as f:
            f.write(f"{'Episode':>8} {'Step':>8} {'Epsilon':>10} {'RewardAvg':>15}"
                    f"{'LengthAvg':>15} {'LossAvg':>15} {'QValueAvg':>15}"
                    f"{'TimeDelta':>15} {'Time':>20}\n"
            )

        self.ep_rewards_plot = os.path.join(save_dir,"reward_plot.jpg")
        self.ep_lengths_plot = os.path.join(save_dir,"length_plot.jpg")
        self.ep_avg_losses_plot = os.path.join(save_dir,"loss_plot.jpg")
        self.ep_avg_qs_plot = os.path.join(save_dir,"q_plot.jpg")

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()

        self.record_time = time.time()
    

    def init_episode(self):
        self.cur_ep_reward = 0.0
        self.cur_ep_length = 0
        self.cur_ep_loss = 0.0
        self.cur_ep_q = 0.0
        self.cur_ep_loss_length = 0
    

    def log_step(self, reward, datas):
        self.cur_ep_reward += reward
        self.cur_ep_length += 1
        if datas[0]:
            self.cur_ep_loss += datas[0]
            self.cur_ep_q += datas[1]
            self.cur_ep_loss_length += 1


    def log_episode(self):
        self.ep_rewards.append(self.cur_ep_reward)
        self.ep_lengths.append(self.cur_ep_length)
        if self.cur_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.cur_ep_loss/self.cur_ep_loss_length,5)
            ep_avg_q = np.round(self.cur_ep_q/self.cur_ep_loss_length,5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()


    def record(self, episode, *datas):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]),3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]),3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]),3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]),3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time-last_record_time,3)

        print(
            f"Episode:{episode} Step:{datas[1]} Epsilon:{np.round(datas[0],3)} Reward-Avg:{mean_ep_reward} Length-Avg:{mean_ep_length}"
            f"Loss-Avg:{mean_ep_loss} Q-Avg:{mean_ep_q} Time-Delta:{time_since_last_record} Time:{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.log_dir, "a") as f:
            f.write(
                f"{episode:8d} {step:8d} {epsilon:10.3f}"
                f"{mean_ep_reward:15.3f} {mean_ep_length:15.3f} {mean_ep_loss:15.3f} {mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f} {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self,f"moving_avg_{metric}"))
            plt.savefig(getattr(self,f"{metric}_plot"))
            plt.clf()


class SACLogger():
    def __init__(self,save_dir):
        self.log_dir = os.path.join(save_dir,"log")
        with open(self.log_dir,"w") as f:
            f.write(f"{'Episode':>8} {'Step':>8} {'RewardAvg':>15}"
                    f"{'LengthAvg':>15} {'LossAvg':>15} {'LossQAvg':>15} {'LossPAvg':>15} {'LossAAvg':>15}"
                    f"{'TimeDelta':>15} {'Time':>20}\n"
            )

        self.ep_rewards_plot = os.path.join(save_dir,"reward_plot.jpg")
        self.ep_lengths_plot = os.path.join(save_dir,"length_plot.jpg")
        self.ep_avg_losses_plot = os.path.join(save_dir,"loss_plot.jpg")
        self.ep_avg_lossqs_plot = os.path.join(save_dir,"loss_q_plot.jpg")
        self.ep_avg_lossps_plot = os.path.join(save_dir,"loss_p_plot.jpg")
        self.ep_avg_lossas_plot = os.path.join(save_dir,"loss_a_plot.jpg")

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_lossqs = []
        self.ep_avg_lossps = []
        self.ep_avg_lossas = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_lossqs = []
        self.moving_avg_ep_avg_lossps = []
        self.moving_avg_ep_avg_lossas = []

        self.init_episode()

        self.record_time = time.time()
    

    def init_episode(self):
        self.cur_ep_reward = 0.0
        self.cur_ep_length = 0
        self.cur_ep_loss = 0.0
        self.cur_ep_loss_q = 0.0
        self.cur_ep_loss_p = 0.0
        self.cur_ep_loss_a = 0.0
        self.cur_ep_loss_length = 0
    

    def log_step(self, reward, datas):
        self.cur_ep_reward += reward
        self.cur_ep_length += 1
        if datas[0]:
            self.cur_ep_loss += datas[0]
            self.cur_ep_loss_q += datas[1]
            self.cur_ep_loss_p += datas[2]
            self.cur_ep_loss_a += datas[3]
            self.cur_ep_loss_length += 1


    def log_episode(self):
        self.ep_rewards.append(self.cur_ep_reward)
        self.ep_lengths.append(self.cur_ep_length)
        if self.cur_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_loss_q = 0
            ep_avg_loss_p = 0
            ep_avg_loss_a = 0
        else:
            ep_avg_loss = np.round(self.cur_ep_loss/self.cur_ep_loss_length,5)
            ep_avg_loss_q = np.round(self.cur_ep_loss_q/self.cur_ep_loss_length, 5)
            ep_avg_loss_p = np.round(self.cur_ep_loss_p/self.cur_ep_loss_length, 5)
            ep_avg_loss_a = np.round(self.cur_ep_loss_a/self.cur_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_lossqs.append(ep_avg_loss_q)
        self.ep_avg_lossps.append(ep_avg_loss_p)
        self.ep_avg_lossas.append(ep_avg_loss_a)

        self.init_episode()


    def record(self, episode, *datas):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]),3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]),3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]),3)
        mean_ep_loss_q = np.round(np.mean(self.ep_avg_lossqs[-100:]),3)
        mean_ep_loss_p = np.round(np.mean(self.ep_avg_lossps[-100:]),3)
        mean_ep_loss_a = np.round(np.mean(self.ep_avg_lossas[-100:]),3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_lossqs.append(mean_ep_loss_q)
        self.moving_avg_ep_avg_lossps.append(mean_ep_loss_p)
        self.moving_avg_ep_avg_lossas.append(mean_ep_loss_a)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time-last_record_time,3)

        print(
            f"Episode:{episode} Step:{datas[1]} Reward-Avg:{mean_ep_reward} Length-Avg:{mean_ep_length}"
            f" Loss-Avg:{mean_ep_loss} LossQ-Avg:{mean_ep_loss_q} LossP-Avg:{mean_ep_loss_p} LossA-Avg:{mean_ep_loss_a} Time-Delta:{time_since_last_record} Time:{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.log_dir, "a") as f:
            f.write(
                f"{episode:8d} {datas[1]:8d} "
                f"{mean_ep_reward:15.3f} {mean_ep_length:15.3f} {mean_ep_loss:15.3f} {mean_ep_loss_q:15.3f} {mean_ep_loss_p:15.3f} {mean_ep_loss_a:15.3f}"
                f" {time_since_last_record:15.3f} {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_lossqs","ep_avg_lossps","ep_avg_lossas"]:
            plt.plot(getattr(self,f"moving_avg_{metric}"))
            plt.savefig(getattr(self,f"{metric}_plot"))
            plt.clf()


class EvalLogger():

    def __init__(self, save_dir):
        self.eval_dir = os.path.join(save_dir, "eval")
        with open(self.eval_dir, "w") as f:
            f.write(f"{'LossAvg':>15} {'RewardAvg':>15} {'QvalAvg':>15}"
                    f" {'LengthAvg':>15} {'TimeAvg':>15} {'Success':>15}\n")

        
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_times = []

        self.success_count = 0
        self.epi_count = 0

        self.init_episode()

    
    def init_episode(self):
        self.cur_ep_reward = 0.0
        self.cur_ep_length = 0
        self.cur_ep_loss = 0.0
        self.cur_ep_q = 0.0
        self.cur_ep_loss_length = 0
        self.record_time = time.time()

    
    def log_step(self, reward, outs):
        loss = outs[0]
        q = outs[1]
        self.cur_ep_reward += reward
        self.cur_ep_length += 1
        if loss:
            self.cur_ep_loss += loss
            self.cur_ep_q += q
            self.cur_ep_loss_length += 1

    
    def log_episode(self, en, success):
        self.ep_rewards.append(self.cur_ep_reward)
        self.ep_lengths.append(self.cur_ep_length)
        if self.cur_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.cur_ep_loss/self.cur_ep_loss_length, 5)
            ep_avg_q = np.round(self.cur_ep_q/self.cur_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        
        ep_time = np.round(time.time() - self.record_time, 3)
        self.ep_times.append(ep_time)

        self.epi_count += 1
        if success:
            self.success_count += 1

        print(f"Epi:{en} Loss:{ep_avg_loss} Reward:{self.cur_ep_reward} Qval:{ep_avg_q}"
              f" Length:{self.cur_ep_length} Time:{ep_time} Success:{success}")
        
        with open(self.eval_dir, "a") as f:
            f.write(f"{ep_avg_loss:15.3f} {self.cur_ep_reward:15.3f} {ep_avg_q:15.3f}"
                    f" {self.cur_ep_length:15.3f} {ep_time:15.3f} {success:15}\n")

        self.init_episode()

    
    def output_eval(self):
        mean_ep_reward = np.round(np.mean(self.ep_rewards), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses), 3)
        mean_ep_q  = np.round(np.mean(self.ep_avg_qs), 3)
        mean_ep_time = np.round(np.mean(self.ep_times), 3)
        
        success_rate = np.round(float(self.success_count)/float(self.epi_count), 3)
        print("Summary")
        print(f"Loss-Avg:{mean_ep_loss} Reward-Avg:{mean_ep_reward} Q-Avg:{mean_ep_q}"
              f" Length-Avg:{mean_ep_length} Time-Avg:{mean_ep_time} Success:{success_rate}({self.success_count}/{self.epi_count})")
        
        with open(self.eval_dir, "a") as f:
            f.write("\nSummary\n")
            f.write(f"{mean_ep_loss:15.3f} {mean_ep_reward:15.3f} {mean_ep_q:15.3f}"
                    f" {mean_ep_length:15.3f} {mean_ep_time:15.3f} {success_rate:15.3f}({self.success_count}/{self.epi_count})\n")

    
class SACEvalLogger():

    def __init__(self, save_dir):
        self.eval_dir = os.path.join(save_dir, "eval")
        with open(self.eval_dir, "w") as f:
            f.write(f"{'LossAvg':>15} {'RewardAvg':>15} {'LossQAvg':>15} {'LossPAvg':>15} {'LossAAvg':>15}"
                    f" {'LengthAvg':>15} {'TimeAvg':>15} {'Success':>15}\n")

        
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_lossqs = []
        self.ep_avg_lossps = []
        self.ep_avg_lossas = []
        self.ep_times = []

        self.success_count = 0
        self.epi_count = 0

        self.init_episode()

    
    def init_episode(self):
        self.cur_ep_reward = 0.0
        self.cur_ep_length = 0
        self.cur_ep_loss = 0.0
        self.cur_ep_loss_q = 0.0
        self.cur_ep_loss_p = 0.0
        self.cur_ep_loss_a = 0.0
        self.cur_ep_loss_length = 0
        self.record_time = time.time()

    
    def log_step(self, reward, outs):
        loss = outs[0]
        loss_q = outs[1]
        loss_p = outs[2]
        loss_a = outs[3]
        self.cur_ep_reward += reward
        self.cur_ep_length += 1
        if loss:
            self.cur_ep_loss += loss
            self.cur_ep_loss_q += loss_q
            self.cur_ep_loss_p += loss_p
            self.cur_ep_loss_a += loss_a
            self.cur_ep_loss_length += 1

    
    def log_episode(self, en, success):
        self.ep_rewards.append(self.cur_ep_reward)
        self.ep_lengths.append(self.cur_ep_length)
        if self.cur_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_loss_q = 0
            ep_avg_loss_p = 0
            ep_avg_loss_a = 0
        else:
            ep_avg_loss = np.round(self.cur_ep_loss/self.cur_ep_loss_length, 5)
            ep_avg_loss_q = np.round(self.cur_ep_loss_q/self.cur_ep_loss_length, 5)
            ep_avg_loss_p = np.round(self.cur_ep_loss_p/self.cur_ep_loss_length, 5)
            ep_avg_loss_a = np.round(self.cur_ep_loss_a/self.cur_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_lossqs.append(ep_avg_loss_q)
        self.ep_avg_lossps.append(ep_avg_loss_p)
        self.ep_avg_lossas.append(ep_avg_loss_a)
        
        ep_time = np.round(time.time() - self.record_time, 3)
        self.ep_times.append(ep_time)

        self.epi_count += 1
        if success:
            self.success_count += 1

        print(f"Epi:{en} Loss:{ep_avg_loss} Reward:{self.cur_ep_reward} LossQ:{ep_avg_loss_q} LossP:{ep_avg_loss_p} LossA:{ep_avg_loss_a}"
              f" Length:{self.cur_ep_length} Time:{ep_time} Success:{success}")
        
        with open(self.eval_dir, "a") as f:
            f.write(f"{ep_avg_loss:15.3f} {self.cur_ep_reward:15.3f} {ep_avg_loss_q:15.3f} {ep_avg_loss_p:15.3f} {ep_avg_loss_a:15.3f}"
                    f" {self.cur_ep_length:15.3f} {ep_time:15.3f} {success:15}\n")

        self.init_episode()

    
    def output_eval(self):
        mean_ep_reward = np.round(np.mean(self.ep_rewards), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses), 3)
        mean_ep_loss_q = np.round(np.mean(self.ep_avg_lossqs), 3)
        mean_ep_loss_p = np.round(np.mean(self.ep_avg_lossps), 3)
        mean_ep_loss_a = np.round(np.mean(self.ep_avg_lossas), 3)
        mean_ep_time = np.round(np.mean(self.ep_times), 3)
        
        success_rate = np.round(float(self.success_count)/float(self.epi_count), 3)
        print("Summary")
        print(f"Loss-Avg:{mean_ep_loss} Reward-Avg:{mean_ep_reward} Loss-Q{mean_ep_loss_q} Loss-P{mean_ep_loss_p} Loss-A{mean_ep_loss_a}"
              f" Length-Avg:{mean_ep_length} Time-Avg:{mean_ep_time} Success:{success_rate}({self.success_count}/{self.epi_count})")
        
        with open(self.eval_dir, "a") as f:
            f.write("\nSummary\n")
            f.write(f"{mean_ep_loss:15.3f} {mean_ep_reward:15.3f} {mean_ep_loss_q:15.3f} {mean_ep_loss_p:15.3f} {mean_ep_loss_a:15.3f}"
                    f" {mean_ep_length:15.3f} {mean_ep_time:15.3f} {success_rate:15.3f}({self.success_count}/{self.epi_count})\n")



    