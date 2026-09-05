import random
import math


from networkx.classes import number_of_edges


def relu(x):
    return max(0,x)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def tanh(x):
    return math.tanh(x)


class SimpleNeuralNetwork:
    def __init__(self, num_of_neurons=15):
        self.weights_output_speed_acceleration = [random.uniform(-1, 1) for _ in range(num_of_neurons)]
        self.weights_output_steering_turning = [random.uniform(-1, 1) for _ in range(num_of_neurons)]
        self.starting_first_bias = random.uniform(-1.0, 1.0)
        self.starting_second_bias = random.uniform(-1.0, 1.0)

    def forward(self, inputs):
        # 1. Accumulate speed signals across all rays
        speed_sum = self.starting_first_bias
        for i in range(len(inputs)):
            speed_sum += inputs[i] * self.weights_output_speed_acceleration[i]
        final_speed = tanh(speed_sum)

        # 2. Accumulate steering signals across all rays
        steer_sum = self.starting_second_bias
        for i in range(len(inputs)):
            steer_sum += inputs[i] * self.weights_output_steering_turning[i]
        final_steer = tanh(steer_sum)

        return [final_speed, final_steer]

if __name__ == "__main__":
    # Create an instance of your network
    brain = SimpleNeuralNetwork(num_of_neurons=15)

    # Fake list of 15 ray distances (e.g. normalized between 0.0 and 1.0)
    chaser_rays = [0.8, 0.2, 0.5, 0.9, 0.1, 0.4, 0.7, 0.3, 0.6, 0.2, 0.8, 0.5, 0.1, 0.9, 0.4]

    outputs = brain.forward(chaser_rays) #  Okay, we somehow need to make it not hardcoded numbers, but rather actually work with game.py
    print(f"Speed Output: {outputs[0]:.4f}")
    print(f"Steer Output: {outputs[1]:.4f}")