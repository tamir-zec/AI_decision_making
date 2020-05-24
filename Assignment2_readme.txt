We used two features: position (height) and velocity.
The method we used to approximate the value function is tiling.
We made a generic format to generate a custom tiling.
In our case we used two grids of size 15*15 with offsets of [0, 0], [0.2, 0.015] accordingly.
This gave us the best results after trying different sizes of tilings, as well as a tiling with 3 features instead of 2 (adding the previous state velocity).
We also used a learning rate decay method, since as we know, Q-learning with linear function approximation doesn't necesseraly converge, and we wanted to prevent "overfitting".