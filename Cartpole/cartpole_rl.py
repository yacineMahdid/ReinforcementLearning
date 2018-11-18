import gym
import random
import numpy as np


# Helper functions
def update_parameters(current_observation, index, parameter):
    if current_observation[index] < parameter[0]:
        parameter[0] = observation[index]
    elif current_observation[index] > parameter[1]:
        parameter[1] = observation[index]
    return parameter

# Map value to a bucket
def discretize(float_value, values, num_buckets):
    increment = (abs(values[0])+values[1])/num_buckets
    for i in range(0, num_buckets):
        if float_value < increment*(i+1) + values[0]:
            return i
# Main

# Parameters
x_position_values = [-0.24, 0.24]
velocity_values = [-2.5, 2.5]
angle_values = [-0.30, 0.30]
angular_velocity_values = [-3.5, 3.5]

x_position_num_bucket = 3
velocity_num_bucket = 3
angle_num_bucket = 7
angular_velocity_num_bucket = 7


number_action = 2
q_table = np.zeros([angle_num_bucket,angular_velocity_num_bucket, number_action])  # First take (2 state and 2 action)

# Hyper-parameters
alpha = 0.2
gamma = 0.1
epsilon = 0.6
max_number_exploration = 200

# Training and testing
env = gym.make('CartPole-v0')
for i_episode in range(600):
    observation = env.reset()
    print("Current episode: " + str(i_episode))
    for t in range(195):
        env.render()

        x_position = discretize(observation[0],x_position_values,x_position_num_bucket)
        velocity = discretize(observation[1],velocity_values,velocity_num_bucket)
        angle = discretize(observation[2],angle_values,angle_num_bucket)
        angular_velocity = discretize(observation[3],angular_velocity_values,angle_num_bucket)

        # Either I choose a random action or use the best thing in the q_table
        if random.uniform(0, 1) < epsilon and i_episode < max_number_exploration:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[angle, angular_velocity])  # Exploit learned values

        # Get the next obsservation
        next_observation, reward, done, info = env.step(action)

        x_position_next = discretize(next_observation[0],x_position_values,x_position_num_bucket)
        velocity_next = discretize(next_observation[1],velocity_values,velocity_num_bucket)
        angle_next = discretize(next_observation[2],angle_values,angle_num_bucket)
        angular_velocity_next = discretize(next_observation[3],angular_velocity_values,angular_velocity_num_bucket)

        old_value = q_table[angle, angular_velocity, action]
        next_max = np.max(q_table[angle_next, angular_velocity_next])

        # Q formula and updating q_table
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[angle, angular_velocity, action] = new_value

        observation = next_observation

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print(q_table)