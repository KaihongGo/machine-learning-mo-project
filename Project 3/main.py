# %%

from Maze import Maze
from Runner import Runner
# from QRobot import QRobot
from torch_py.MinDQNRobot import MinDQNRobot as Robot  # PyTorch版本

"""  Deep Qlearning 算法相关参数： """

epoch = 20  # 训练轮数
maze_size = 5  # 迷宫size
training_per_epoch = int(maze_size * maze_size * 2)

""" 使用 DQN 算法训练 """

maze = Maze(maze_size=maze_size)
robot = Robot(maze)

print(robot.maze.reward)  # 输出最小值选择策略的reward值

"""开启金手指，获取全图视野"""
robot.memory.build_full_view(maze=maze)
runner = Runner(robot)
runner.run_training(epoch, training_per_epoch)
runner.plot_results()

# """Test Robot"""
# robot.reset()
# for _ in range(25):
#     a, r = robot.test_update()
#     print("action:", a, "reward:", r)
#     if r == maze.reward["destination"]:
#         print("success")
#         break

# 生成训练过程的gif图, 建议下载到本地查看；也可以注释该行代码，加快运行速度。
# runner.generate_gif(filename="results/dqn_size10.gif")

# %%
