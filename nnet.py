import math
import random

class SimpleNeuralNetwork:
    def __init__(self, input_nodes=30, hidden_nodes=16, output_nodes=2):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Layer 1 Weights (Input -> Hidden) & Biases
        self.w1 = [[random.uniform(-1, 1) for _ in range(hidden_nodes)] for _ in range(input_nodes)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_nodes)]

        # Layer 2 Weights (Hidden -> Output) & Biases
        self.w2 = [[random.uniform(-1, 1) for _ in range(output_nodes)] for _ in range(hidden_nodes)]
        self.b2 = [random.uniform(-1, 1) for _ in range(output_nodes)]

    def relu(self, x):
        # Hidden layer activation: passes positive signals, blocks negative ones
        return max(0.0, x)

    def tanh(self, x):
        # Output layer activation: clamps results strictly between -1.0 and 1.0.
        return math.tanh(x)

    def forward(self, inputs):
        #Feeds 30 inputs through hidden layer to produce 2 movement outputs.
        if len(inputs) != self.input_nodes:
            # Fallback padding/truncation if input length mismatches during testing
            inputs = (inputs + [1.0] * self.input_nodes)[:self.input_nodes]

        # 1. Input Layer -> Hidden Layer
        hidden_outputs = []
        for j in range(self.hidden_nodes):
            total = self.b1[j]
            for i in range(self.input_nodes):
                total += inputs[i] * self.w1[i][j]
            hidden_outputs.append(self.relu(total))

        # 2. Hidden Layer -> Output Layer
        final_outputs = []
        for k in range(self.output_nodes):
            total = self.b2[k]
            for j in range(self.hidden_nodes):
                total += hidden_outputs[j] * self.w2[j][k]
            final_outputs.append(self.tanh(total))

        return final_outputs  # Returns [move_speed, turn_angle]

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        # Randomly tweaks weights/biases to evolve better behaviors over generations.
        # Mutate W1 & B1
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes):
                if random.random() < mutation_rate:
                    self.w1[i][j] += random.gauss(0, mutation_strength)

        for j in range(self.hidden_nodes):
            if random.random() < mutation_rate:
                self.b1[j] += random.gauss(0, mutation_strength)

        # Mutate W2 & B2
        for j in range(self.hidden_nodes):
            for k in range(self.output_nodes):
                if random.random() < mutation_rate:
                    self.w2[j][k] += random.gauss(0, mutation_strength)

        for k in range(self.output_nodes):
            if random.random() < mutation_rate:
                self.b2[k] += random.gauss(0, mutation_strength)