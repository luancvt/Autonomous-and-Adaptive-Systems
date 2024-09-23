import numpy as np
import matplotlib.pyplot as plt

from keras import layers, Model, ops
from keras.api.models import load_model
from keras.api.optimizers import Adam
import tensorflow as tf

DROPOUT=0.2
TRAIN_EPS=30000

IMAGE_HEIGHT=64
IMAGE_WIDTH=64
CHANNELS=3
FRAMES=4

LEARNING_RATE=0.0001

class REINFORCE_CNN():
    def __init__(self, enviroment):
        self.env = enviroment
        self.action_space = enviroment.action_space.n
        self.state_space = (FRAMES,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

        # Remove unecessary actions
        self.action_space = enviroment.action_space.n - 6

        # Game states 4 images.
        self.images_states = np.zeros(self.state_space)

        self.state_history = []
        self.action_log_prob_history = []
        self.reward_history = []

        # Game stats
        self.scores = []
        self.average_score_history = []

        # Optimisers
        self.opt_policy = Adam(learning_rate=LEARNING_RATE)
        self.opt_value = Adam(learning_rate=LEARNING_RATE)

        # Create policy and value network models
        self.policy_model, self.value_model = self.get_models(action_space = self.action_space)


    def get_models(self, action_space):
        # Need to make 2 network structures, one for the policy and one for the value.
        inputs = layers.Input(shape=self.state_space)

        # Shared Convolutional Base
        x = layers.Conv3D(filters=32, kernel_size=(2, 8, 8), strides=(1, 4, 4), activation='relu', data_format='channels_last')(inputs)
        x = layers.Conv3D(filters=64, kernel_size=(2, 4, 4), strides=(1, 2, 2), activation='relu')(x)
        x = layers.Conv3D(filters=64, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = layers.Dropout(DROPOUT)(x)

        # Policy Network
        action_output = layers.Dense(action_space, activation='softmax')(x)

        # Value Network
        value_output = layers.Dense(1, activation='linear')(x)

        # Define the models
        policy_model = Model(inputs=inputs, outputs=action_output)
        value_model = Model(inputs=inputs, outputs=value_output)

        return policy_model, value_model


    def step(self, action):
        next_state_frame, reward, done, info = self.env.step(action)
        self.images_states = np.roll(self.images_states, 1, axis = 0)
        self.images_states[0,:,:,:] = next_state_frame
        image_states = self.images_states

        return image_states, reward, done, info


    def train(self):
        for episode in range(TRAIN_EPS):
            # Reset necessary variables
            self.images_states = np.zeros(self.state_space)
            state_frame_reset = self.env.reset()

            for frame in range(FRAMES):
                self.images_states[frame, :, :, :] = state_frame_reset
            state = self.images_states

            self.state_history = []
            self.action_log_prob_history = []

            self.reward_history = []
            score = 0
            steps = 0
            done = False

            while not done:
                predicted_action_probs = self.policy_model.predict(state.reshape(1,4,64,64,3), batch_size=32, verbose=0)[0]
                chosen_action = np.random.choice(self.action_space, p=predicted_action_probs)

                next_state, reward, done, info = self.step(chosen_action)

                self.store(state, predicted_action_probs, chosen_action, reward)

                state = next_state
                score += reward
                steps += 1

                if done:
                    self.scores.append(score)
                    current_average_score = sum(self.scores)/len(self.scores)
                    self.average_score_history.append(current_average_score)

                    self.save()
                    print(f'Model saved at episode {episode}')
                    print()
                    print(f'episode: {episode}/{TRAIN_EPS}, score: {score}, average: {current_average_score}, steps: {steps}')
                    print(' ')
                    self.update_networks()
                    self.save_stats()

        self.env.close()


    def update_networks(self):
        # Shape data for training
        states = tf.convert_to_tensor(self.state_history)
        action_probs = tf.convert_to_tensor(np.reshape(self.action_log_prob_history,(len(self.action_log_prob_history),1)))
        rewards = tf.convert_to_tensor(np.reshape(self.reward_history,(len(self.reward_history),1)))
        
        # Gamma is 1 so we do not calculate discounted reward
        with tf.GradientTape() as tape_policy:
            pred_values = tf.convert_to_tensor(self.policy_model(states, training=True))
            delta = tf.subtract(rewards, pred_values)
            policy_loss = tf.multiply(action_probs, delta)
            policy_loss = tf.multiply(policy_loss,-1)
        
        with tf.GradientTape() as tape_value:
            pred_values = tf.convert_to_tensor(self.value_model(states, training=True))
            error = tf.subtract(rewards, pred_values)
            value_loss = tf.pow(error,2)

        # Fittting the network with new weights given gradients
        grads_policy = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)
        grads_value = tape_value.gradient(value_loss, self.value_model.trainable_variables)

        self.opt_policy.apply_gradients(zip(grads_policy, self.policy_model.trainable_variables))
        self.opt_value.apply_gradients(zip(grads_value, self.value_model.trainable_variables))


    def test(self):
        self.load('policy_model.keras','value_model.keras')
        
        for episode in range(100):
            # Reset necessary variables
            self.images_states = np.zeros(self.state_space)
            state_frame_reset = self.env.reset()

            for frame in range(FRAMES):
                self.images_states[frame, :, :, :] = state_frame_reset
            state = self.images_states

            self.state_history = []
            self.action_log_prob_history = []

            self.reward_history = []
            score = 0
            steps = 0
            done = False

            while not done:
                predicted_action_probs = self.policy_model.predict(state.reshape(1,4,64,64,3), batch_size=32, verbose=0)[0]
                chosen_action = np.random.choice(self.action_space, p=predicted_action_probs)

                next_state, reward, done, _ = self.step(chosen_action)

                steps += 1

                self.store(state, predicted_action_probs, chosen_action, reward)

                state = next_state
                score += reward

                if done:
                    break

            print(f'episode: {episode}/{TRAIN_EPS}, score: {score}, steps: {steps}')

        self.env.close()


    def store(self, images_states, action_prob, action, reward):
        self.state_history.append(images_states)
        self.action_log_prob_history.append(ops.log(action_prob[action]))
        self.reward_history.append(reward)


    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        axs[0].plot(self.average_score_history, label='Average Score', color='orange')
        axs[0].set_ylabel('Score')
        axs[0].set_title('Average Score')

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def save(self):
        self.policy_model.save('policy_model.keras')
        self.value_model.save('value_model.keras')

    
    def load(self, policy_file_name, value_file_name):
        self.policy_model = load_model(policy_file_name)
        self.value_model = load_model(value_file_name)


    def save_stats(self):
        np.save('average_score_history.npy', self.average_score_history)
        return
    