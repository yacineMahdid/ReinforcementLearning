import gym


def update_parameters(current_observation, index, parameter):
    if current_observation[index] < parameter[0]:
        parameter[0] = observation[index]
    elif current_observation[index] > parameter[1]:
        parameter[1] = observation[index]
    return parameter


env = gym.make('CartPole-v0')
x_position = [1000, -1000]
velocity = [1000, -1000]
angle = [1000, -1000]
angular_velocity = [1000, -1000]
for i_episode in range(20):
    observation = env.reset()
    for t in range(195):
        env.render()

        # Updating the observations min and max
        x_position = update_parameters(observation, 0, x_position)
        velocity = update_parameters(observation, 1, velocity)
        angle = update_parameters(observation, 2, angle)
        angular_velocity = update_parameters(observation, 3, angular_velocity)

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


# Printing the min and max values
print("X Position: " + str(x_position))
print("Velocity: " + str(velocity))
print("Angle: " + str(angle))
print("Angular velocity: " + str(angular_velocity))