import numpy as np
import torch
import os

from PIL import Image
import cv2

from logger import EvalLogger, SACEvalLogger

class Evaluate():
    
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.agent_name = args.agent

        self.save_dir = args.save_dir

        self.episode_num = args.eval_episode_num
        
        self.eval_record = args.eval_record

        if args.checkpoint:
            self.agent.load(args.checkpoint)
        else:
            print("Note: CheckPoint Dir is not specified!")
    

    def process(self):
        self.env.reset()

        if not self.eval_record:
            if self.agent_name != "SAC":
                logger = EvalLogger(self.save_dir)
            else:
                logger = SACEvalLogger(self.save_dir)
        else:
            img_shape = np.array(self.env.render(mode='rgb_array')).shape
            codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        for i in range(self.episode_num):
            state = self.env.reset()

            if self.eval_record:
                video = cv2.VideoWriter(os.path.join(self.save_dir,"{0:03d}.mp4".format(i)),codec, 30.0, (img_shape[1],img_shape[0]))

            # screen = self.env.render(mode='rgb_array')[:,:,[2,1,0]]
            # img = Image.fromarray(screen)
            # img.save(os.path.join(self.save_dir,"{0:3d}.jpg".format(i)))

            success = False
            while True:
                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)

                is_end = self.env.is_end(done,info)
                is_success = self.env.is_success(info)

                outs = self.agent.eval(state, next_state, action, reward, done, info)
                state = next_state

                if not self.eval_record:
                    screen = self.env.render()
                    logger.log_step(reward, outs)
                else:
                    screen = self.env.render(mode='rgb_array')[:,:,[2,1,0]]
                    video.write(screen)


                if is_success:
                    success = True

                if is_end:
                    break
            
            if not self.eval_record:
                logger.log_episode(i, success)
            else:
                video.release()

        if not self.eval_record:
            logger.output_eval()

        self.env.close()