import os
import pickle
import tensorflow as tf
import numpy as np
import gym
import logz
import inspect


#============================================================================================#
# Utilities
#============================================================================================#
def build_mlp(input_placeholder, output_size, layers, size, activation=tf.nn.relu, output_activation=None):
	"""
		build a feedforward neuron network for policy

		arguments:
			input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

		return:
			output placeholder of the neuron network
	"""
	out=input_placeholder
	for _ in range(layers):
		out=tf.layers.dense(out, units=size, activation=activation)
	out=tf.layers.dense(out, units=output_size, activation=output_activation)
	return out

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_func)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


#============================================================================================#
# Behavioral Cloning
#============================================================================================#
class BC_Agent(object):
	"""
		only for continuous action space
	"""
	def __init__(self, env_name, ob_dim, act_dim, layers, size, learning_rate, max_path_length):
		self.env_name = env_name
		self.ob_dim=ob_dim
		self.act_dim=act_dim
		self.layers=layers
		self.size=size
		self.learning_rate=learning_rate
		self.max_path_length=max_path_length


	def build_graph(self):
		"""
			build the forward graph

		"""
		self.sy_obs=tf.placeholder(shape=[None, self.ob_dim], dtype=tf.float32)
		self.sy_labels=tf.placeholder(shape=[None, self.act_dim], dtype=tf.float32)
		self.sy_logits=build_mlp(self.sy_obs, self.act_dim, self.layers, self.size)
		self.loss=tf.losses.mean_squared_error(labels=self.sy_labels, predictions=self.sy_logits)
		self.update_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


	def init_tf_sess(self):
		"""
			initialize the global variables

		"""
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def update_parameters(self, obs, labels):
		"""
			update the parameters of the neuron networks

			return:
				loss

		"""
		_ ,loss=self.sess.run([self.update_op,self.loss], feed_dict={self.sy_obs: obs, self.sy_labels: labels})
		return loss


	def evaluation(self, env, n_rollouts):
		"""
			evaluate the model and save observations for DAgger 

			arguments：
				env: the environment object
				n_rollouts: perform n rollouts to compute average return

			return: [rewards, path_length, obs]
				rewards: the rewards of multiple roll-outs
				path_length: the length of each rollout
				obs: obs for DAgger
		"""
		obs, rewards, path_length=[],[],[]
		for i in range(n_rollouts):
			ob=env.reset()
			steps=0
			reward=0
			while(True):
				obs.append(ob)
				ac=self.sess.run(self.sy_logits, feed_dict={self.sy_obs: ob.reshape(1,-1)})
				ob, rew, done, _ =env.step(ac[0])
				reward+=rew
				steps+=1
				if done or steps>self.max_path_length:
					break
			rewards.append(reward)
			path_length.append(steps)

		return rewards,path_length,obs
		

def train_func(exp_name, env_name, n_epoch, batch_size, learning_rate, layers, size, n_rollouts):
	"""
		train the agent with expert_data

        expert_data = {'observations': np.array(observations),
               'actions': np.array(actions)}

        arguments:
  			layers: hidden layers of the neuron networks
  			size: number of units of the hidden layers
        	

	"""

	# load training data
	PROJECT_ROOT=os.path.realpath(__file__)
	data_path=os.path.join(os.path.dirname(PROJECT_ROOT), 'expert_data', env_name+'.pkl')
	expert_data=pickle.load(open(data_path, 'rb'))
	expert_data['actions']=np.squeeze(expert_data['actions'])
	data_size=expert_data['observations'].shape[0]

	# make the gym environment
	env=gym.make(env_name)
	ob_dim=env.observation_space.shape[0]
	act_dim=env.action_space.shape[0]
	max_path_length=env.spec.max_episode_steps

	agent=BC_Agent(env_name, ob_dim, act_dim, layers, size, learning_rate, max_path_length)

	# build graph
	agent.build_graph()
	agent.init_tf_sess()

	# setup logger
	logdir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", exp_name+".txt")
	setup_logger(logdir, locals())

	# training loop
	for epoch in range(n_epoch):
		print("************** Epoch %i **************"%epoch)
		losses=[]
		for _ in range(int(data_size/batch_size)):
			shuffle_idx=np.random.randint(low=0, high=data_size, size=batch_size)
			obs=expert_data['observations'][shuffle_idx]
			labels=expert_data['actions'][shuffle_idx]
			loss=agent.update_parameters(obs, labels)
			losses.append(loss)

		# log metrics per epoch and save to the logger file
		rewards, path_length, _ =agent.evaluation(env, n_rollouts)
		logz.log_tabular("Iteration", epoch)
		logz.log_tabular("AverageLoss", np.mean(losses))
		logz.log_tabular("StdLoss", np.std(losses))
		logz.log_tabular("AverageReturn", np.mean(rewards))
		logz.log_tabular("MaxReturn", np.max(rewards))
		logz.log_tabular("MinReturn", np.min(rewards))
		logz.log_tabular("StdReturn", np.std(rewards))
		logz.log_tabular("AverageLen", np.mean(path_length))
		logz.log_tabular("MaxLen", np.max(path_length))
		logz.log_tabular("MinLen", np.min(path_length))
		logz.log_tabular("StdLen", np.std(path_length))
		logz.dump_tabular()




if __name__=="__main__":	
	train_func(exp_name='BC_Humanoid-v2', env_name='Humanoid-v2',n_epoch=100, batch_size=32, learning_rate=1e-3, layers=2, size=64, n_rollouts=20)
