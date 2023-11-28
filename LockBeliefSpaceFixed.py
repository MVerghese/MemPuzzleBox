import numpy as np
import time 
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
import warnings
import PuzzleBoxEnv

gamma = .99
np.set_printoptions(suppress=True)
# import warnings
# warnings.filterwarnings("error")
np.seterr(all='raise')
ent_max = 15.
ent_min = 1.


def compute_entropy(P):
	Ptemp = P[np.where(P>0)]
	return(-1*np.sum(np.multiply(Ptemp,np.log(Ptemp))))


def compute_KL_Divergence(P,Q,gamma = .99, enforce_sum = False):

	Q = Q[np.where(P>0)]
	P = P[np.where(P>0)]
	

	Q = np.clip(Q,1-gamma,gamma)

	if enforce_sum:
		assert(np.allclose(np.sum(Ptotal,axis=0),1.))
		# assert(np.allclose(np.sum(Qtotal,axis=0),1.))
	return(np.sum(np.multiply(P,np.log(P/Q))))

def compute_ce(P,Q):
	print("Entropy",compute_entropy(P))
	print("KL Div",compute_KL_Divergence(P,Q))

	return(compute_entropy(P)+compute_KL_Divergence(P,Q))

def bayesian_inference(ph,peh,penoth):
	try:
		out = 1/(1+(1/ph - 1)*penoth/peh)
	except FloatingPointError:
		out = 0
		print(ph,peh,penoth)
		1/0
	return(out)

def p_or(X,Y):
	assert(X <= 1)
	assert(Y <= 1)
	return(1-((1-X)*(1-Y)))

def normalize_belief_state(belief_state):
	for b in range(belief_state.shape[2]-1):
		z = 1 - np.prod(1-belief_state[:-1,:,b]) + belief_state[-1,0,b]
		belief_state[-1,0,b] /= z
		# print((1 - np.prod(1-belief_state[:-1,:,b]))/z+belief_state[-1,0,b])
		z1 = np.prod(1-belief_state[:-1,:,b])/(1 - 1/z + np.prod(1-belief_state[:-1,:,b])/z)
		# print('z1',z1)
		# print(1 - np.prod(1-belief_state[:-1,:,b])/z1+belief_state[-1,0,b])
		# print(1-belief_state[:-1,:,b])
		z2 = z1**(1/(belief_state.shape[0]-2)/2)
		# print('z2',z2)
		# print(1-np.prod((1-belief_state[:-1,:,b])/z2) + belief_state[-1,0,b])
		belief_state[:-1,:,b] = 1 - 1/z2 + belief_state[:-1,:,b]/z2
		belief_state[b,:,b] = 0.
		if np.any(belief_state[:-1,:,b] < 0):
			belief_state[:-1,:,b] = 0.
			belief_state[-1,:,b] = 1.
	return(belief_state)

def print_belief_state(belief_state,label=''):
	print(label)
	print(np.around(belief_state[:,0,:,0],5))
	print(np.around(belief_state[:,1,:,0],5))
	print(np.around(belief_state[:,0,:,1],5))
	print(np.around(belief_state[:,1,:,1],5))




def compute_posterior_divergence(belief_state, current_state, action):
	# print(1 - belief_state[np.arange(0,current_state.shape[0]),current_state,np.ones(current_state.shape[0],dtype=int)*action])
	rel_belief_state = belief_state[:,:,action,current_state[action]] #dim: NxS
	indep_prob = rel_belief_state[action,0]
	rel_belief_state[action,1:] = 0
	
	total_divergence = 0

	cur_lock_probs = rel_belief_state[np.arange(0,current_state.shape[0]),current_state] #dim: N
	cur_lock_probs[action] = 0
	p_moves = 1-np.sum(cur_lock_probs)

	# print("Action: ",action)
	# print("PMoves: ",p_moves)
	
	#MOVES
	if p_moves > 1e-5:
		new_belief_state = np.copy(rel_belief_state)
		new_belief_state[np.arange(0,current_state.shape[0]),current_state] = 0.
		new_belief_state[action,0] = indep_prob
		
		try:
			new_belief_state /= np.sum(new_belief_state)
		except FloatingPointError:
			print(current_state)
			print(action)
			print(p_moves)
			print(rel_belief_state)
			print(new_belief_state)
		
		# print("MOVES")
		# print(rel_belief_state)
		# print(new_belief_state)
		total_divergence += p_moves*compute_KL_Divergence(rel_belief_state,new_belief_state)

	#DOESNT MOVE
	if p_moves < 1-1e-5:
		new_belief_state = np.copy(rel_belief_state)
		new_belief_state[np.arange(0,current_state.shape[0]),1-current_state] = 0.
		new_belief_state[action,0] = 0
		# print(action)
		# print(p_moves)
		# print(rel_belief_state)
		# print(new_belief_state)
		new_belief_state /= np.sum(new_belief_state)
		# print("DOESNT MOVE")
		# print(rel_belief_state)
		# print(new_belief_state)
		total_divergence += (1-p_moves)*compute_KL_Divergence(rel_belief_state,new_belief_state)


	return(total_divergence)


def compute_posterior_actual(belief_state, current_state, action, success):
	rel_belief_state = belief_state[:,:,action,current_state[action]] #dim: NxS
	indep_prob = rel_belief_state[action,0]
	rel_belief_state[action,1:] = 0
	# print("belief space update thinks action state is",current_state[action])
	
	if success:
		rel_belief_state[np.arange(0,current_state.shape[0]),current_state] = 0.
		rel_belief_state[action,0] = indep_prob
		rel_belief_state /= np.sum(rel_belief_state)


	else:
		rel_belief_state[np.arange(0,current_state.shape[0]),1-current_state] = 0.
		rel_belief_state[action,0] = 0
		rel_belief_state /= np.sum(rel_belief_state)

	belief_state[:,:,action,current_state[action]] = rel_belief_state
	
	return(belief_state)	


def update_belief_state(belief_state, current_state, action, success):
	new_belief_state = compute_posterior_actual(belief_state, current_state, action, success)
	return(new_belief_state)



def compute_info_gain(belief_state,current_state):
	info_gain = np.zeros(current_state.shape[0])
	for action in range(current_state.shape[0]):
		info_gain[action] += compute_posterior_divergence(belief_state, current_state, action)

	return(info_gain)



def argmax(arr):
	if np.sum(np.max(arr) == arr) <= 1:
		return(np.argmax(arr))
	else:
		idxs = np.where(np.max(arr) == arr)[0]
		# print(idxs)
		idx = np.random.choice(idxs)
		return(idx)

def argmin(arr):
	if np.sum(np.min(arr) == arr) <= 1:
		return(np.argmin(arr))
	else:
		idxs = np.where(np.min(arr) == arr)[0]
		# print(idxs)
		idx = np.random.choice(idxs)
		return(idx)

def hash_state(state):
	weights = np.array([2**i for i in range(0,len(state))])
	return(np.dot(state,weights))

def info_max_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	info_gain = compute_info_gain(belief_state, current_state)
	info_gain = np.clip(info_gain,0,None)
	info_gain[np.where(info_gain < (1-gamma)*1.01)] = 0
	# if prev_action >= 0:
	# 	info_gain[prev_action] = 0

	# print(info_gain)
	# print(action_prob)
	action = argmax(info_gain)
	# print(action_prob[action])
	return(action,info_gain)

def random_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	action_prob = np.random.uniform(0,1,size = current_state.shape[0])
	if prev_action >= 0:
		action_prob[prev_action] = 0
	action = argmax(action_prob)
	return(action,1/np.sum(current_state.shape[0]))

def greedy_policy(belief_state,current_state,state_hash,prev_action,**kwargs):

	lock_probs = np.multiply(belief_state,1-current_state.reshape((current_state.shape[0],1)))
	unlock_probs = 1 - lock_probs
	prob_unlocked =  np.prod(unlock_probs,axis=0)
	action_prob = prob_unlocked+current_state*-100
	action = np.argmax(action_prob)
	return(action,action_prob)

def balanced_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	alpha = .5
	if "alpha" in kwargs:
		alpha = kwargs["alpha"]

	info_action,info_action_probs = info_max_policy(belief_state,current_state,**kwargs)
	plan_action,plan_action_probs = greedy_policy(belief_state,current_state,**kwargs)
	action_scores = alpha*info_action_probs + (1-alpha)*plan_action_probs

	action = np.argmax(action_scores)
	return(action,action_scores)

def constant_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	return(0,1)

def novelty_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	q_vals = compute_q_novelty(current_state,belief_state,state_hash)
	if prev_action >= 0:
		q_vals[prev_action] = 0
	action = argmax(q_vals)
	if np.all(q_vals == 0):
		return(action,np.zeros(current_state.shape[0]))
	else:
		return(action,q_vals/np.sum(q_vals))

def compute_actuation_prob(belief_state,current_state,full_state=None):
	actuation_prob = np.zeros(current_state.shape)
	for action in range(current_state.shape[0]):
		rel_belief_state = belief_state[:,:,action,current_state[action]] #dim: NxS
		indep_prob = rel_belief_state[action,0]
		rel_belief_state[action,1:] = 0
	

		cur_lock_probs = rel_belief_state[np.arange(0,current_state.shape[0]),current_state] #dim: N
		cur_lock_probs[action] = 0
		p_moves = 1-np.sum(cur_lock_probs)
		actuation_prob[action] = p_or(p_moves,indep_prob)
	return(actuation_prob)

def exploration_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	alpha = .5
	if "alpha" in kwargs:
		alpha = kwargs["alpha"]

	info_action,info_action_scores = info_max_policy(belief_state,current_state,state_hash,prev_action,**kwargs)
	novl_action,novl_action_scores = novelty_policy(belief_state,current_state,state_hash,prev_action,**kwargs)
	h = compute_entropy(belief_state)
	weighting = (h-ent_min)/(ent_max-ent_min)

	action_probs = info_action_scores*(weighting)+novl_action_scores*(1-weighting)
	if prev_action >= 0:
		action_probs[prev_action] = 0
	return(argmax(action_probs),action_probs)

def optimal_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	goal_state = current_state.shape[0] - 1

	if "goal_state" in kwargs:
		goal_state = kwargs["goal_state"]
	actuation_prob_func = compute_actuation_prob
	if "actuation_prob_func" in kwargs:
		actuation_prob_func = kwargs['actuation_prob_func']

	q_vals = compute_q(current_state,belief_state,goal_state,actuation_prob_func)
	# if prev_action >= 0:
	# 	q_vals[prev_action] = 0
	action = argmax(q_vals)
	if np.all(q_vals == 0):
		return(action,np.zeros(current_state.shape[0]))
	else:
		return(action,q_vals/np.sum(q_vals))

def dual_policy(belief_state,current_state,state_hash,prev_action,**kwargs):
	ent_min = 1.
	ent_max = 15.
	if "ent_max" in kwargs:
		ent_max = kwargs["ent_max"]
	if "ent_min" in kwargs:
		ent_min = kwargs["ent_min"]
	info_action,info_action_scores = info_max_policy(belief_state,current_state,state_hash,prev_action,**kwargs)
	optm_action,optm_action_scores = optimal_policy(belief_state,current_state,state_hash,prev_action,**kwargs)
	# print("INFO SCORES: ", info_action_scores)
	# print("OPTM SCORES: ", optm_action_scores)
	h = compute_entropy(belief_state)
	weighting = (h-ent_min)/(ent_max-ent_min)

	action_probs = info_action_scores*(weighting)+optm_action_scores*(1-weighting)


	#UNCOMMENT FOR DISASSEMBLY
	# action_probs[np.where(current_state == 1)] = -1
	
	action = argmax(action_probs)
	return(action,action_probs)


class search_node:
	def __init__(self, state, bf, value = 0.):
		self.state = state
		self.connections = [None]*bf
		self.ancestor = None
		self.probabilities = np.zeros(bf)
		self.value = value

class dijkstra_search:
	def __init__(self,start_state,bf,belief_state,actuation_prob_func,full_state=None,decay=.90):
		self.bf = bf
		self.nodes = [None]*2**bf
		start_state = start_state.astype(int)
		start_node = search_node(start_state,bf,value=1.)
		self.nodes[hash_state(start_state)] = start_node
		self.start_state = start_state
		self.open_set = []
		self.open_set.append(start_node)
		self.closed_set = []
		self.belief_state = belief_state
		self.decay = decay
		self.actuation_prob_func = actuation_prob_func
		self.full_state = full_state

	def expand_node(self,node):
		current_state = node.state
		if np.any(self.full_state):
			current_full_state = np.copy(self.full_state)
			current_full_state[:current_state.shape[0]] = current_state
		else:
			current_full_state = []

		actuation_probs = self.actuation_prob_func(self.belief_state,current_state,full_state = current_full_state)
		node.probabilities = actuation_probs
		for action in range(current_state.shape[0]):
			proposed_state = current_state.copy()
			proposed_state[action] = 1 - proposed_state[action]
			if self.nodes[hash_state(proposed_state)]:
				node.connections[action] = self.nodes[hash_state(proposed_state)]
				if node.value*node.probabilities[action]*self.decay > node.connections[action].value:
					node.connections[action].value = node.value*node.probabilities[action]*self.decay
					node.connections[action].ancestor = node

					if node.connections[action] not in self.open_set:
						self.open_set.append(node.connections[action])
					if node.connections[action] in self.closed_set:
						self.closed_set.remove(node.connections[action])

			else:
				new_node = search_node(proposed_state,self.bf,node.value*node.probabilities[action]*self.decay)
				node.connections[action] = new_node
				new_node.ancestor = node
				self.nodes[hash_state(proposed_state)] = new_node
				self.open_set.append(new_node)

		self.open_set.remove(node)
		self.closed_set.append(node)

	def return_best_node(self,nodes):
		best_node = nodes[0]
		for node in nodes:
			if node.value > best_node.value:
				best_node = node
		return(best_node)

	def is_goal_state(self,state,goal_states):
		ret = False
		for goal_state in goal_states:
			if np.all(state == goal_state):
				ret = True
				break
		return(ret)


	def search(self,goal_states,verbose=False):
		goal = None
		counter = 0
		while len(self.open_set) > 0:
			node = self.return_best_node(self.open_set)
			# print(node.state,node.value)
			if self.is_goal_state(node.state, goal_states):
				goal = node
				# print(goal.value)
				break
			else:
				counter += 1
				self.expand_node(node)
			if counter > 300:
				if verbose:
					print(len(self.closed_set))
					for node in self.closed_set:
						print(node.state)
					print("PLANNER TIMED OUT")
				break
				# 1/0
		if verbose:
			print(counter)

		if goal == None:
			if verbose:
				print("search failed")
			return(False,0,[],[])
		else:
			action_sequence = []
			state_sequence = [goal.state]
			final_cost = goal.value
			cur_node = goal
			while not self.is_goal_state(cur_node.state, [self.start_state]):
				optimal = -1
				for action in range(self.bf):
					if cur_node.ancestor.connections[action] == cur_node:
						optimal = action
				action_sequence = [optimal] + action_sequence
				cur_node = cur_node.ancestor
				state_sequence = [cur_node.state] + state_sequence
			return(True,final_cost, state_sequence,action_sequence)

def compute_q(state,belief_state,goal_state,actuation_prob_func,full_state = None):
	n = state.shape[0]
	goals = []
	for i in range(0,n**2): 
		chars = [c for c in bin(i)[2:]]
		temp_state = np.flip(np.array(chars).astype(int))
		padded_state = np.zeros(n,dtype=int)
		padded_state[:temp_state.shape[0]] = temp_state
		if padded_state[goal_state] == 1:
			goals.append(padded_state)
	if np.any(full_state):
		current_full_state = np.copy(full_state)
		current_full_state[:state.shape[0]] = state
	else:
		current_full_state = []
	actuation_probs = actuation_prob_func(belief_state,state,current_full_state)
	q_vals = np.zeros(n)
	for action in range(n):
		proposed_state = state.copy()
		proposed_state[action] = 1 - proposed_state[action]
		search_obj = dijkstra_search(proposed_state,n,belief_state,actuation_prob_func,full_state = full_state)
		res, final_cost, state_sequence,action_sequence = search_obj.search(goals)
		# print("Action: ",actuation_probs[action],final_cost)
		q_vals[action] = actuation_probs[action]*final_cost

	return(q_vals)

def compute_q_novelty(state,belief_state,state_hash):
	n = state.shape[0]
	goals = []
	for i in range(n**2):
		if state_hash[i] == 0:
			chars = [c for c in bin(i)[2:]]
			goals.append(np.flip(np.array(chars).astype(int)))
	actuation_probs = compute_actuation_prob(belief_state,state)
	q_vals = np.zeros(n)
	for action in range(n):
		proposed_state = state.copy()
		proposed_state[action] = 1 - proposed_state[action]
		search_obj = dijkstra_search(proposed_state,n,belief_state)
		res, final_cost, state_sequence,action_sequence = search_obj.search(goals)
		q_vals[action] = actuation_probs[action]*final_cost

	return(q_vals)

def init_belief_state(num_components, num_states, indep_prior = .7):
	belief_state = np.ones((num_components,num_states,num_components,num_states))
	for i in range(num_components):
		belief_state[i,1:,i,:] = 0
		for j in range(num_states):
			belief_state[:,:,i,j] *= (1-indep_prior)/((num_components-1)*num_states)
			belief_state[i,0,i,j] = indep_prior
	return(belief_state)




def eval_policy(policy,size,environment='original', prior=None,runs = 1000,timeout=100,stop_on_completion = True, policy_args={},verbose=False):
	total = 0
	distribution_divergences = np.zeros((runs,timeout))

	for run in range(runs):
		env = PuzzleBoxEnv.LockEnv(environment,size,2)
		# print_belief_state(config)
		# 1/0
		config = env.config
		if verbose:
			plt.imshow(np.vstack((np.hstack((config[:,0,:,0],config[:,1,:,0])),
								  np.hstack((config[:,0,:,1],config[:,1,:,1])))),
				       vmin=0,vmax=1)
			plt.show()
		current_state = np.zeros(size,dtype=int)
		if np.any(prior == None):
			belief_state = init_belief_state(size,config.shape[1],indep_prior = .7)
		else:
			belief_state = prior

		
		success, state, reward, done = env.reset()
		current_state = np.copy(current_state)
		counter = 0
		# fig = plt.figure()
		# ax = fig.gca()
		state_hash = np.zeros(2**size,dtype=int)
		state_hash[hash_state(current_state)] = 1
		
		if verbose:
			bs = plt.imshow(np.vstack((np.hstack((belief_state[:,0,:,0],belief_state[:,1,:,0])),
								  	   np.hstack((belief_state[:,0,:,1],belief_state[:,1,:,1])))),
				       	   vmin=0,vmax=1)
			# print(np.around(belief_state[:,0,:],5))
			# print(np.around(belief_state[:,1,:],5))
			plt.draw()
			plt.pause(0.05)
		completed = False
		prev_action = -1
		while reward < 1 and counter < timeout:
			if verbose:
				print("Iterations: ", counter)
			# print(policy_args)
			action, confidence = policy(belief_state,current_state,state_hash,prev_action,**policy_args)
			if verbose:
				# pass
				print(confidence)
				print(action)
				
			# print("Pre step", current_state)
			success, new_state, reward, done = env.step(action)
			prev_action = action
			# print("post step", current_state)
			
			if verbose:
				# pass
				print(current_state)
				print(success)
				print(new_state)

			belief_state = update_belief_state(belief_state, current_state, action, success)
			current_state = np.copy(new_state)
			state_hash[hash_state(current_state)] = 1
			if verbose:
				pass
				# print(belief_state)

				print_belief_state(belief_state,'Sim Loop')

			distribution_divergence = compute_KL_Divergence(np.clip(belief_state,0.01,0.99),np.clip(config,0.01,0.99))
			# distribution_divergence = compute_entropy(np.clip(belief_state[:size,:,:size],0.01,0.99))
			distribution_divergences[run,counter] = distribution_divergence
			if verbose:
				print("Distribution Error: ", distribution_divergence)
				print("Belief State Entropy: ", compute_entropy(belief_state))
				print("Percent of states visited: ", np.sum(state_hash)/state_hash.shape[0])
			if verbose:
				bs.set_data(np.vstack((np.hstack((belief_state[:,0,:,0],belief_state[:,1,:,0])),
								  	   np.hstack((belief_state[:,0,:,1],belief_state[:,1,:,1])))))
				plt.draw()
				plt.pause(0.05)

			counter += 1
			if not stop_on_completion:
				if done and not completed:
					print("COMPLETED")
					completed = True
				reward = 0

			if verbose:
				time.sleep(.001)
		print(counter)



		total += counter

	divergence_over_t_mean = []
	divergence_over_t_var = []
	if verbose:
		bs.set_data(np.vstack((np.hstack((belief_state[:,0,:,0],belief_state[:,1,:,0])),
							   np.hstack((belief_state[:,0,:,1],belief_state[:,1,:,1])))))
		plt.show()


	for i in range(timeout):
		if np.all(distribution_divergences[:,i] == 0):
			break
		divergence_over_t_mean.append(np.mean(distribution_divergences[np.where(distribution_divergences[:,i] > 0)[0],i]))
		divergence_over_t_var.append(np.var(distribution_divergences[np.where(distribution_divergences[:,i] > 0)[0],i]))
	return(total/runs, np.array(divergence_over_t_mean),np.array(divergence_over_t_var),belief_state)

def calculate_mean_var(data):
	divergence_over_t_mean = []
	divergence_over_t_var = []
	for i in range(data.shape[1]):
		if np.all(data[:,i] == 0):
			break
		divergence_over_t_mean.append(np.mean(data[np.where(data[:,i] > 0)[0],i]))
		divergence_over_t_var.append(np.var(data[np.where(data[:,i] > 0)[0],i]))
	return(total/runs, np.array(divergence_over_t_mean),np.array(divergence_over_t_var))

def adapt_prior(prior,idxs,entropy = .1):
	new_prior = np.zeros((idxs.shape[0],prior.shape[1],idxs.shape[0],prior.shape[3]))
	for i in range(idxs.shape[0]):
		temp = prior[idxs[i]]
		# print(temp.shape)
		new_prior[i,:,:,:] = temp[:,idxs,:]
		new_prior += entropy
		new_prior[i,1:,i,:] = 0

	for action in range(idxs.shape[0]):
		for state in range(new_prior.shape[3]):
			rel_belief_state = new_prior[:,:,action,state]
			rel_belief_state /= np.sum(rel_belief_state)
			new_prior[:,:,action,state] = rel_belief_state

		
	return(new_prior)



def train_agent(agent,num_train_envs,eps_per_env = 1,env_type = 'train',dump_graphs=False):
	print("Training agent...")
	train_counts = np.zeros((eps_per_env,num_train_envs))
	# print(num_train_envs)
	for ep in range(eps_per_env):
		print("ep: ",ep)
		for i in range(num_train_envs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = i,env_permutation=ep)
			# print(env.get_object_ids())
			agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = False)
			# print(env.get_component_locations())
			success, state, reward, done = env.reset()
			counter = 0
			while not done:
				action = agent.act(state)
				success, new_state, reward, done = env.step(action)
				agent.update_policy_info(state,success)
				state = new_state
				counter += 1
			train_counts[ep,i] = counter
			agent.save_graph()
	if dump_graphs:
		agent.dump_graphs()

	return(np.mean(train_counts,axis=0))

def eval_agent(agent,num_test_envs,eps_per_env = 1,env_type = 'test',dump_graphs=False):
	print("Evaluating agent...")
	test_counts = np.zeros((eps_per_env,num_test_envs))
	for ep in range(eps_per_env):
		print("ep: ",ep)
		for i in range(num_test_envs):
			env = PuzzleBoxEnv.LockEnv(env_type,5,2,env_index = i,env_permutation=ep)
			agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
			success, state, reward, done = env.reset()
			counter = 0
			while not done:
				action = agent.act(state,ent_max = 200)
				success, new_state, reward, done = env.step(action)
				agent.update_policy_info(state,success)
				state = new_state
				counter += 1
			test_counts[ep,i] = counter
			agent.save_graph()
	if dump_graphs:
		agent.dump_graphs()

	return(np.mean(test_counts,axis=0))



class Structured_Agent_Dist:

	def __init__(self,max_num_components,dim = 2):
		# max_num_components: int, maximum number of components in the environment
		# dim int, environment dimension, 2 for planar puzzleboxes and 3 for 3D puzzleboxes
		self.prior_graphs = []
		self.prior_cids = []
		self.prior_dists = []

		self.index_mat = np.arange(max_num_components**2).reshape((max_num_components,max_num_components))
		self.component_interaction_spaces = []
		self.prior_component_interactions = []
		if dim == 2:
			self.dist_size = 4
		else:
			self.dist_size = 9
		for i in range(max_num_components**2):
			self.component_interaction_spaces.append(np.zeros((0,self.dist_size)))
			self.prior_component_interactions.append(np.zeros((0,2,2)))

	def dump_graphs(self,graph_filename = 'prior_graphs.npy',cids_filename = 'prior_cids.npy',dists_filename = 'prior_dists.npy'):
		np.save(graph_filename,self.prior_graphs, allow_pickle = True)
		np.save(cids_filename,self.prior_cids, allow_pickle = True)
		np.save(dists_filename,self.prior_dists, allow_pickle = True)

	def load_graphs(self,graph_filename = 'prior_graphs.npy',cids_filename = 'prior_cids.npy',dists_filename = 'prior_dists.npy'):
		self.prior_graphs = np.load(graph_filename, allow_pickle = True)
		self.prior_cids = np.load(cids_filename, allow_pickle = True)
		self.prior_dists = np.load(dists_filename, allow_pickle = True)


	def init_policy(self,num_components,cids,dists,goal_state,use_priors = True, priors_k = 5, prior_entropy = 0.01):
		# num_components: int, number of components in the environment
		# cids: np.array (num_components,), id of each component in the environment
		# dists: np.array (num_components, 4), each components physical location expresses as (x,y,sin(theta),cos(theta))
		# goal_state: int, index of the component that must be toggled for success.
		# use_priors: bool, whether or not to use priors, should be true for evaluation
		# priors_k: int, number of priors to use for each component relationship
		# prior_entropy: float, entropy to add to each prior component relationship
		self.cids = cids
		self.dists = dists
		self.goal_state = goal_state
		self.state_hash = np.zeros(2**num_components)
		self.belief_state = np.zeros((num_components,2,num_components,2))
		
		if use_priors:
			# Iterate over every pairwise relationship of components
			for i in range(num_components):
				for j in range(num_components):
					# Compute the relative transform between the two components
					rel_dist = (dists[j] - dists[i]).reshape((1,self.dist_size))
					# Find the right bucket based on the two component ids
					inter_idx = self.index_mat[self.cids[i],self.cids[j]]
					# Compute the similarity between the relative transform and all of the prior transforms
					diffs = self.component_interaction_spaces[inter_idx] - rel_dist
					rel_dist_distances = np.sqrt(np.sum(diffs**2,axis=1))
					max_dist = np.max(rel_dist_distances)
					dist_copy = np.copy(rel_dist_distances)
					relevant_graph_idxs = []
					for k in range(min(len(self.prior_cids),priors_k)):
						idx = argmin(dist_copy)
						relevant_graph_idxs.append(idx)
						dist_copy[idx] = max_dist+1

					# Compute the probability of a locking interaction as the average of the relevant prior component relationships
					component_relationship = np.sum(self.prior_component_interactions[inter_idx][relevant_graph_idxs,:,:],axis=0)
					component_relationship /= len(relevant_graph_idxs)

					self.belief_state[i,:,j,:] = component_relationship

			# add entropy to the prior belief state
			self.belief_state += prior_entropy
			for i in range(num_components):
				self.belief_state[i,0,i,:] += 0
				self.belief_state[i,1:,i,:] = 0
			# normalize belief state
			for action in range(num_components):
				for state in range(self.belief_state.shape[3]):
					rel_belief_state = self.belief_state[:,:,action,state]
					rel_belief_state /= np.sum(rel_belief_state)
					self.belief_state[:,:,action,state] = rel_belief_state


			# print_belief_state(self.belief_state,"new_belief_state")

		else:
			self.belief_state = init_belief_state(num_components, 2)


		self.prev_action = -1


	def act(self,state,explore_only = False,verbose = False, ent_max = 15., ent_min = 1.):
		# state: np.array (num_components,), current state of the environment
		# explore_only: bool, whether or not to only use the exploration policy
		# verbose: bool, whether or not to print out information about the policy
		# ent_max: float, maximum entropy value to use for the dual policy's weighing
		# ent_min: float, minimum entropy value to use for the dual policy's weighing
		# returns: int, which component id to attempt to toggle (from 0 state to 1 state or from 1 state to 0 state)
		if not explore_only:
			action, confidence = dual_policy(self.belief_state,state,self.state_hash,self.prev_action,goal_state = self.goal_state,ent_max = ent_max,ent_min = ent_min)
		else:
			action, confidence = info_max_policy(self.belief_state,state,self.state_hash,self.prev_action)
		if verbose:
			print(action, confidence)
		self.prev_action = action
		return(action)

	def update_policy_info(self,old_state,success):
		# old_state: np.array (num_components,), previous state of the environment
		# success: bool, whether or not the previous action was successful
		self.state_hash[hash_state(old_state)] = 1
		self.belief_state = update_belief_state(self.belief_state, old_state, self.prev_action, success)

	def save_graph(self):
		self.prior_graphs.append(self.belief_state)
		self.prior_cids.append(self.cids)
		self.prior_dists.append(self.dists)


		# add the current belief state to the prior experience library
		for i in range(self.belief_state.shape[0]):
			for j in range(self.belief_state.shape[2]):
				idx = self.index_mat[self.cids[i],self.cids[j]]
				self.component_interaction_spaces[idx] = np.vstack((self.component_interaction_spaces[idx],(self.dists[j] - self.dists[i]).reshape((1,self.dist_size))))
				self.prior_component_interactions[idx] = np.vstack((self.prior_component_interactions[idx],self.belief_state[i,:,j,:].reshape((1,2,2))))


def rect(pos):
    r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor="k", linewidth=1.2)
    plt.gca().add_patch(r)

def config_plot(config,title):
	rc('text', usetex=True)
	rc('axes', linewidth=2)
	rc('font', weight='bold')
	rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
	rcParams["legend.loc"] = 'lower right'
	plt.imshow(config,vmin=0,vmax=1)
	ax = plt.gca()
	ax.set_xticks(np.arange(5))
	ax.set_xticklabels([r'\textbf{W1}',r'\textbf{S1}',r'\textbf{D1}',r'\textbf{S2}',r'\textbf{D2}'])
	ax.set_yticks(np.arange(5))
	ax.set_yticklabels([r'\textbf{W1}',r'\textbf{S1}',r'\textbf{D1}',r'\textbf{S2}',r'\textbf{D2}'])

	x, y = np.meshgrid(np.arange(5),np.arange(5))
	pos = np.c_[x.flatten(),y.flatten()]
	for p in pos:
		rect(p)


	plt.title(title)
	# plt.show()





def main():
	np.random.seed(1)
	agent = Structured_Agent_Dist(5,dim=2)
	train_agent(agent,9,eps_per_env=5,dump_graphs=True)
	test_counts = eval_agent(agent,3,eps_per_env=3,dump_graphs=True)
	print(test_counts)
	1/0


	# for i in [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]:
	# 	env = PuzzleBoxEnv.CompEnv(env_index = i,jamming=0.0)
	# 	print(env.get_object_ids())
	# 	locations = np.array([[24.0653,-7.0341,-1,-0.0239],
	# 						  [18.0313,-7.2389,-1,-0.0230],
	# 						  [10.4267,-6.9789,-1,-0.0187],
	# 						  [11.4360,0.7323,0.0090,-1],
	# 						  [11.4134,7.5497,0.0568,-0.9984]])			
	# 	agent.init_policy(8,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = False)
	# 	success, state, reward, done = env.reset()
	# 	counter = 0
	# 	while not done:
	# 		action = agent.act(state)
	# 		print(counter)
	# 		# print(state)
	# 		# print(action)
	# 		success, new_state, reward, done = env.step(action)
	# 		agent.update_policy_info(state,success)
	# 		state = new_state
	# 		counter += 1
	# 	print(counter)
	# 	agent.save_graph()

	# print("TESTING")

	# for i in [1]:
	# 	env = PuzzleBoxEnv.CompEnv(env_index = i,jamming=0.0)
	# 	print(env.get_object_ids())
	# 	locations = np.array([[24.0653,-7.0341,-1,-0.0239],
	# 						  [18.0313,-7.2389,-1,-0.0230],
	# 						  [10.4267,-6.9789,-1,-0.0187],
	# 						  [11.4360,0.7323,0.0090,-1],
	# 						  [11.4134,7.5497,0.0568,-0.9984]])			
	# 	agent.init_policy(8,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
	# 	success, state, reward, done = env.reset()
	# 	counter = 0
	# 	while not done:
	# 		action = agent.act(state, ent_max = 200)
	# 		print(counter)
	# 		print(state)
	# 		print(action)
	# 		success, new_state, reward, done = env.step(action)
	# 		agent.update_policy_info(state,success)
	# 		state = new_state
	# 		counter += 1
	# 	print(counter)
	# 	agent.save_graph()
	# 1/0

	np.random.seed(0)
	
	agent = Structured_Agent_Dist(3)
	print('TRAINING')
	total_iters = 0
	for j in range(1):
		for i in range(9):
			env = PuzzleBoxEnv.LockEnv('all',5,2,env_index = i,jamming=0.0)
			print(env.get_object_ids())
			locations = np.array([[24.0653,-7.0341,-1,-0.0239],
								  [18.0313,-7.2389,-1,-0.0230],
								  [10.4267,-6.9789,-1,-0.0187],
								  [11.4360,0.7323,0.0090,-1],
								  [11.4134,7.5497,0.0568,-0.9984]])	
			if j  < 1:
				agent.init_policy(5,env.get_object_ids(),locations,env.get_goal_state(),use_priors = False)
			else:
				agent.init_policy(5,env.get_object_ids(),locations,env.get_goal_state(),use_priors = False)
			success, state, reward, done = env.reset()
			counter = 0
			while not done:
				action = agent.act(state)
				print(counter)
				print(state)
				print(action)
				success, new_state, reward, done = env.step(action)
				agent.update_policy_info(state,success)
				state = new_state
				counter += 1
				total_iters += 1
				if total_iters > 107*1.5:
					print("Time Up")
					break
			print(counter)
			agent.save_graph()
	# 1/0
	# print('TRAINING RESULTS')
	# for i in range(9):
	# 	env = PuzzleBoxEnv.LockEnv('train',5,2,env_index = i)
	# 	print(env.get_object_ids())
	# 	agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
	# 	success, state, reward, done = env.reset()
	# 	counter = 0
	# 	while not done:
	# 		action = agent.act(state,ent_max = 200)
	# 		success, new_state, reward, done = env.step(action)
	# 		agent.update_policy_info(state,success)
	# 		state = new_state
	# 		counter += 1
	# 	print(counter)
	# 	# agent.save_graph()
	# print('TESTING')
	# for i in range(3):
	# 	env = PuzzleBoxEnv.LockEnv('test',5,2,env_index = i)
	# 	print("Current CIDS", env.get_object_ids())
	# 	agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
	# 	success, state, reward, done = env.reset()
	# 	counter = 0
	# 	while not done:
	# 		# print("Action Selection")
	# 		action = agent.act(state,ent_max = 200)
	# 		success, new_state, reward, done = env.step(action)
	# 		# print("Information update")
	# 		agent.update_policy_info(state,success)
	# 		state = new_state
	# 		counter += 1
	# 	print("Steps: ", counter)
		# agent.save_graph()
	# env = PuzzleBoxEnv.LockEnv('random',5,2)
	# print("Current CIDS", env.get_object_ids())
	# agent.init_policy(5,env.get_object_ids(),env.get_goal_state(),use_priors = True, priors_k=1)
	print("TESTING")
	np.random.seed(5)
	# for i in range(1):
	# 	counter_total = 0
	# 	for i in range(5):
	# 		env = PuzzleBoxEnv.LockEnv('all',5,2,env_index = 8,jamming=0.0)
	# 		print("Current CIDS", env.get_object_ids())
	# 		locations = np.array([[23.0993,-0.1519,0.6930,0.7210],
	# 							  [16.5110,-4.7502,0.6710,-0.7415],
	# 							  [12.8176,-9.8099,0.0359,-0.9993],
	# 							  [10.6856,-4.1045,0.0253,-1.0000],
	# 							  [11.1199,1.9743,0.0763,-0.9971],])
	# 		agent.init_policy(5,env.get_object_ids(),locations,env.get_goal_state(),priors_k = 5, use_priors = True)
	# 		success, state, reward, done = env.reset()
	# 		counter = 0
	# 		config_plot(agent.belief_state[:,0,:,0],r'\textbf{Prior}')
	# 		plt.savefig('Prior.eps')
	# 		counter = 0
	# 		while not done:
	# 			# print("Action Selection")
	# 			# print("")
	# 			action = agent.act(state,ent_max = 200)
	# 			# print(counter)
	# 			# print(state)
	# 			# print(action)
	# 			success, new_state, reward, done = env.step(action)
	# 			config_plot(agent.belief_state[:,0,:,0],r'\textbf{Action: %i}'%(counter+1))
	# 			# plt.savefig('Action'+str(counter+1)+'.eps')
	# 			# print("Information update")
	# 			agent.update_policy_info(state,success)
	# 			state = new_state
	# 			counter += 1
	# 		print("Steps: ", counter)
	# 		counter_total += counter
	# 		# x = input("")
	# 	# agent.save_graph()
	# 	print("Average:", counter_total/5)

	for i in range(1):
		counter_total = 0
		for i in range(5):
			env = PuzzleBoxEnv.LockEnv('all',5,2,env_index = 2,jamming=0.0)
			print("Current CIDS", env.get_object_ids())
			locations = np.array([[24.0653,-7.0341,-1,-0.0239],
								  [18.0313,-7.2389,-1,-0.0230],
								  [10.4267,-6.9789,-1,-0.0187],
								  [11.4360,0.7323,0.0090,-1],
								  [11.4134,7.5497,0.0568,-0.9984]])	
			agent.init_policy(5,env.get_object_ids(),locations,env.get_goal_state(),priors_k = 5, use_priors = True)
			success, state, reward, done = env.reset()
			counter = 0
			config_plot(agent.belief_state[:,0,:,0],r'\textbf{Prior}')
			plt.savefig('Prior.eps')
			counter = 0
			while not done:
				# print("Action Selection")
				# print("")
				action = agent.act(state,ent_max = 200)
				# print(counter)
				# print(state)
				# print(action)
				success, new_state, reward, done = env.step(action)
				config_plot(agent.belief_state[:,0,:,0],r'\textbf{Action: %i}'%(counter+1))
				# plt.savefig('Action'+str(counter+1)+'.eps')
				# print("Information update")
				agent.update_policy_info(state,success)
				state = new_state
				counter += 1
			print("Steps: ", counter)
			counter_total += counter
			# x = input("")
		# agent.save_graph()
		print("Average:", counter_total/5)

	#avg 7.6

	# np.random.seed(1)
	# for i in range(1):
	# 	env = PuzzleBoxEnv.LockEnv('all',5,2,env_index = 8)
	# 	print("Current CIDS", env.get_object_ids())
	# 	locations = np.array([[23.0993,-0.1519,0.6930,0.7210],
	# 						  [16.5110,-4.7502,0.6710,-0.7415],
	# 						  [12.8176,-9.8099,0.0359,-0.9993],
	# 						  [10.6856,-4.1045,0.0253,-1.0000],
	# 						  [11.1199,1.9743,0.0763,-0.9971],])
	# 	agent.init_policy(5,env.get_object_ids(),env.get_component_locations(),env.get_goal_state(),use_priors = True)
	# 	success, state, reward, done = env.reset()
	# 	counter = 0
	# 	config_plot(agent.belief_state[:,0,:,0],r'\textbf{Prior}')
	# 	plt.savefig('Prior.eps')
	# 	counter = 0
	# 	while not done:
	# 		# print("Action Selection")
	# 		action = agent.act(state,ent_max = 200)
	# 		print("")
	# 		print(counter)
	# 		print(state)
	# 		print(action)
	# 		success, new_state, reward, done = env.step(action)
	# 		config_plot(agent.belief_state[:,0,:,0],r'\textbf{Action: %i}'%(counter+1))
	# 		# plt.savefig('Action'+str(counter+1)+'.eps')
	# 		# print("Information update")
	# 		agent.update_policy_info(state,success)
	# 		state = new_state
	# 		counter += 1
	# 	print("Steps: ", counter)
	# 	# x = input("")
	# 	agent.save_graph()
	1/0


	

	# plt.imshow(np.hstack((prior[:,0,:],prior[:,1,:])),vmin=0,vmax=1)
	# plt.show()
	# prior = init_belief_state(5,2)

	# search_obj = dijkstra_search(np.array([0,0,0,0,0]),5,prior)
	# goals = []
	# for i in range(16,32):
	# 	chars = [c for c in bin(i)[2:]]
	# 	goals.append(np.flip(np.array(chars).astype(int)))
	# print(goals)
	# # goals = [np.array([1,1,1,1,1])]
	# # print(goals)
	# res, final_cost, state_sequence,action_sequence = search_obj.search(goals)
	# print(res,final_cost,state_sequence,action_sequence)
	# print(compute_entropy(prior))
	# print(compute_q(np.array([0,0,0,0,0]),prior))
	# 1/0





	info_avg_actions, info_divergence_mean, info_divergence_var, belief_state = eval_policy(dual_policy,5,environment = 'random',prior=None,runs=1,timeout=200,stop_on_completion = True, verbose=True,policy_args={"alpha":.5})
	print(info_avg_actions)
	1/0
	# belief_state = init_belief_state(5,2)
	# print(belief_state.shape)
	plt.imshow(np.vstack((np.hstack((belief_state[:,0,:,0],belief_state[:,1,:,0])),
								  np.hstack((belief_state[:,0,:,1],belief_state[:,1,:,1])))),
				       vmin=0,vmax=1)
	plt.show()

	transfer_task = 1

	tt_ids = PuzzleBoxEnv.config_dict[PuzzleBoxEnv.TT1s[transfer_task]][1]

	# print(tt_ids)
	# print(new_prior.shape)
	new_prior = adapt_prior(belief_state,tt_ids,entropy=0.01)

	plt.imshow(np.vstack((np.hstack((new_prior[:,0,:,0],new_prior[:,1,:,0])),
								  np.hstack((new_prior[:,0,:,1],new_prior[:,1,:,1])))),
				       vmin=0,vmax=1)
	plt.show()


	print("Transfer Task")
	info_avg_actions, info_divergence_mean, info_divergence_var, belief_state = eval_policy(dual_policy,4,environment=PuzzleBoxEnv.TT1s[transfer_task],prior=new_prior,runs=1,timeout=200,stop_on_completion = True, verbose=True,policy_args={"alpha":.5})

	



	1/0
	print('INFO')
	info_avg_actions, info_divergence_mean, info_divergence_var, _ = eval_policy(info_max_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False)
	print('RAND')
	rand_avg_actions, rand_divergence_mean, rand_divergence_var, _ = eval_policy(random_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False)
	print('NOVL')
	novl_avg_actions, novl_divergence_mean, novl_divergence_var, _ = eval_policy(exploration_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False,policy_args={"alpha":.9})
	print('OPTM')
	optm_avg_actions, optm_divergence_mean, optm_divergence_var, _ = eval_policy(dual_policy,5,stop_on_completion = True, runs=100,timeout=100,verbose=False,policy_args={"alpha":.9})


	print(info_avg_actions,rand_avg_actions, novl_avg_actions, optm_avg_actions)

	info_divergence_var /= 3
	rand_divergence_var /= 3
	novl_divergence_var /= 3
	optm_divergence_var /= 3



	fig = plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean,c='b',label="Info_Max")
	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean-info_divergence_var,c='b',alpha=.5)
	plt.plot(np.arange(len(info_divergence_mean)),info_divergence_mean+info_divergence_var,c='b',alpha=.5)
	plt.fill_between(np.arange(len(info_divergence_mean)),info_divergence_mean-info_divergence_var,info_divergence_mean+info_divergence_var,facecolor='b',alpha=.5)
	
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean,c='r',label="Random")
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean-rand_divergence_var,c='r',alpha=.5)
	plt.plot(np.arange(len(rand_divergence_mean)),rand_divergence_mean+rand_divergence_var,c='r',alpha=.5)
	plt.fill_between(np.arange(len(rand_divergence_mean)),rand_divergence_mean-rand_divergence_var,rand_divergence_mean+rand_divergence_var,facecolor='r',alpha=.5)

	plt.plot(np.arange(len(novl_divergence_mean)),novl_divergence_mean,c='g',label="Exploration")
	plt.plot(np.arange(len(novl_divergence_mean)),novl_divergence_mean-novl_divergence_var,c='g',alpha=.5)
	plt.plot(np.arange(len(novl_divergence_mean)),novl_divergence_mean+novl_divergence_var,c='g',alpha=.5)
	plt.fill_between(np.arange(len(novl_divergence_mean)),novl_divergence_mean-novl_divergence_var,novl_divergence_mean+novl_divergence_var,facecolor='g',alpha=.5)

	plt.plot(np.arange(len(optm_divergence_mean)),optm_divergence_mean,c='c',label="Dual")
	plt.plot(np.arange(len(optm_divergence_mean)),optm_divergence_mean-optm_divergence_var,c='c',alpha=.5)
	plt.plot(np.arange(len(optm_divergence_mean)),optm_divergence_mean+optm_divergence_var,c='c',alpha=.5)
	plt.fill_between(np.arange(len(optm_divergence_mean)),optm_divergence_mean-optm_divergence_var,optm_divergence_mean+optm_divergence_var,facecolor='c',alpha=.5)

	plt.legend()
	plt.xlabel("Actions")
	plt.ylabel("KL Divergence")
	plt.show()


	

	1/0

	sizes = np.arange(3,10,1)
	info_max = []
	random = []
	balanced_3 = []
	balanced_5 = []
	balanced_7 = []
	balanced_9 = []



	for size in sizes:
		print(size)
		info_max.append(eval_policy(info_max_policy,size,runs=1000))
		random.append(eval_policy(random_policy,size,runs=1000))
		print("Info max improvement: ",float(info_max[-1])/float(random[-1]))
		print("Optimal improvement: ",float(size)/float(random[-1]))

		balanced_3.append(eval_policy(balanced_policy,size,policy_args={"alpha":.3}))
		balanced_5.append(eval_policy(balanced_policy,size,policy_args={"alpha":.5}))
		balanced_7.append(eval_policy(balanced_policy,size,policy_args={"alpha":.7}))
		balanced_9.append(eval_policy(balanced_policy,size,policy_args={"alpha":.9}))
		print("balanced_3 improvement: ",float(balanced_3[-1])/float(random[-1]))
		print("balanced_5 improvement: ",float(balanced_5[-1])/float(random[-1]))
		print("balanced_7 improvement: ",float(balanced_7[-1])/float(random[-1]))
		print("balanced_9 improvement: ",float(balanced_9[-1])/float(random[-1]))


	fig = plt.figure(figsize=(12,8))
	plt.plot(sizes,np.array(info_max),c='b',label='info_max')
	plt.plot(sizes,np.array(random),c='r',label='random')
	plt.plot(sizes,np.array(balanced_3),c='c',label='balanced .3')
	plt.plot(sizes,np.array(balanced_5),c='m',label='balanced .5')
	plt.plot(sizes,np.array(balanced_7),c='y',label='balanced .7')
	plt.plot(sizes,np.array(balanced_9),c='k',label='balanced .9')


	plt.plot(sizes,sizes,c='g',label='optimal')

	plt.legend()
	plt.xlabel("Number of locks")
	plt.ylabel("Average actions")
	plt.show()








if __name__ == '__main__':
	main()
