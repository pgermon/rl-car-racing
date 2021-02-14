# rl-car-racing

Authors: [Paul Germon](https://github.com/pgermon) and [Johan Chataignier](https://github.com/JohanChataigne)  

This projects aims to train a DQNAgent on the Gym OpenAI Car-Racing environment. We created a custom environment in `car_racing_v1.py` in order to make the action space discrete instead of continuous.

## Installation

1 - Get the repository

`git clone https://github.com/pgermon/rl-car-racing
cd rl-car-racing/
`

2 - Install it, using Conda for example (use Python >= 3.6)  
`
conda create --name myenv python=3.6
conda activate myenv
pip install -r requirements.txt
`

3 - Launch a training  
`
python dqn_agent.py
`
or using the notebook `dqn_agent.ipynb`