we used a gaussian distribution for the policy function with a fixed but decaying sigma(variance).
the mean of the distribution is defined by the linear combination of our features and theta (the weights vector).
the policy feature vector we choose to use is of boolean features:
	1.1 if velocity > 0 then 1 otherwise 0
	2.if velocity < 0 then 1 otherwise 0
	3.if velocity = 0 then 1 otherwise 0
	4.if velocity*location > 0 then 1 otherwise 0
	5.if velocity*location < 0 then 1 otherwise 0
we also used linear state_action value function approximation with the following features:
	1. location
	2. velocity
	3. action value
we update the weights vector - 'w' with TD(0) method.