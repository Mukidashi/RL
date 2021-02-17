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
    

    def log_step(self, reward, loss, q):
        self.cur_ep_reward += reward
        self.cur_ep_length += 1
        if loss:
            self.cur_ep_loss += loss
            self.cur_ep_q += q
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


    def record(self, episode, epsilon, step):
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
            f"Episode:{episode} Step:{step} Epsilon:{np.round(epsilon,3)} Reward-Avg:{mean_ep_reward} Length-Avg:{mean_ep_length}"
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

    
    def log_step(self, reward, loss, q):
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

    




    