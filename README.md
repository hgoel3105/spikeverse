# BCS Spikeverse: Reinforcement Learning with Spiking Neural Networks

This repository contains the code, assignments, and research documentation for our project on integrating Spiking Neural Networks (SNNs) with Deep Reinforcement Learning (RL). The project was developed under the Brain and Cognitive Science (BCS) Club at IIT Kanpur.

## Project Background

Traditional Artificial Neural Networks (ANNs), such as Deep Q-Networks (DQNs), rely on continuous floating-point activations and dense matrix multiplications. While effective, they carry significant computational and energy costs when executed on standard hardware. Spiking Neural Networks compute asynchronously using discrete binary events (spikes), offering an alternative geared toward low-power neuromorphic hardware.

During our experiments, we observed that training deep SNNs directly via surrogate gradient backpropagation in sparse-reward RL environments (such as Atari Breakout) is computationally expensive and prone to training instability. To overcome these issues, we adopted an ANN-to-SNN conversion methodology:

1. **Continuous Baseline Training:** We train a continuous convolutional/dense Deep Q-Network on `BreakoutNoFrameskip-v4` using temporal frame differencing and frame stacking.
2. **Weight Transfer:** The learned weights from the continuous network are loaded into a Leaky Integrate-and-Fire (LIF) spiking neural network.
3. **Layer-Wise Weight Scaling:** To account for reduced firing rates across deeper layers, we apply deterministic scaling factors (10x for the first layer and 100x for the second layer).
4. **Rate Encoding Evaluation:** Incoming pixel states are converted into binary spike trains over a 500 ms simulation window using Poisson rate coding.

## Repository Layout

```text
BCS-Spikeverse/
│
├── src/                                   # Core implementation scripts
│   ├── config.py                          # Hyperparameters and network settings
│   ├── Environment.py                     # Atari Breakout environment wrapper and frame processing
│   ├── Neuron.py                          # Leaky Integrate-and-Fire (LIF) neuron models
│   ├── Architectures.py                   # Continuous DQN and temporal SNN architectures
│   ├── Agent.py                           # Experience replay buffer and agent execution logic
│   ├── Train.py                           # Training and weight conversion pipeline
│   └── Test.py                            # Evaluation script for benchmark metrics
│
├── docs/                                  # Formal project documentation
│   ├── Documentation.pdf                  # 16-page technical report and methodology breakdown
│   └── BCS_Spikeverse_Resume_Report.md    # Summary of architectural benchmarks and metrics
│
├── assignments/                           # Coursework and exploratory phase modules
│   ├── Harshil_spikeverse_assignment_01.ipynb # Q-learning, GridWorld, and CartPole DQN implementation
│   ├── PER_Phase2.ipynb                   # Prioritized Experience Replay using a binary SumTree
│   └── Assignment_3/                      # Direct supervised SNN training on the MNIST dataset
│
├── weights/                               # Model checkpoints (.pth files)
├── plots/                                 # Training evaluation curves and histograms
├── requirements.txt                       # Python dependencies
└── README.md                              # Project overview
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/hgoel3105/spikeverse.git
cd spikeverse
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

The main execution scripts are located in the `src/` directory. When executing commands from the repository root, the scripts automatically locate the stored weights and export plots to their respective folders.

### Running the Training & Conversion Pipeline
To execute the fine-tuning workflow, load the pre-trained continuous weights, scale them into the spiking network, and generate training plots:

```bash
python src/Train.py
```
This script writes the converted spiking network weights to `weights/snn_model_weights.pth` and saves the training metrics chart to `plots/SNN_Fine_Tune.png`.

### Evaluating the Spiking Agent
To run inference evaluation across 100 test episodes without updating weights:

```bash
python src/Test.py
```
This script outputs the mean and peak reward scores to the console and generates a reward distribution histogram in `plots/SNN_binary_test_rewards.png`.

## Core Implementation Details

* **Adaptive LIF Neurons:** Implemented in `Neuron.py`, the neurons incorporate dynamic threshold adaptation (`theta_plus = 0.05`) to regulate firing rates during prolonged stimulation.
* **Prioritized Experience Replay:** Implemented in `PER_Phase2.ipynb`, using a binary SumTree data structure to sample transitions proportional to temporal difference error with O(log N) complexity.
* **State Preprocessing:** Implemented in `Environment.py`, game frames undergo 50% center cropping, downsampling to 80x80 arrays, and consecutive frame differencing to capture object velocity.

## Project Mentors and Affiliation

* **Organization:** Brain and Cognitive Science (BCS) Club, IIT Kanpur
* **Mentors:** Gaurav Rampuria, Kshitiz Tyagi, Saubhagya Pandey
