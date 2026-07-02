# BCS Spikeverse — Reinforcement Learning with Spiking Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-Atari%20RL-0081a7?style=for-the-badge)
![Neuromorphic](https://img.shields.io/badge/Domain-Neuromorphic%20Computing-00b4d8?style=for-the-badge)

A state-of-the-art machine learning framework exploring the intersection of **Deep Reinforcement Learning (RL)** and **Neuromorphic Computing**. Developed as part of the Brain and Cognitive Science (BCS) Club initiative, this project bridges continuous Artificial Neural Networks (ANNs) and event-driven Spiking Neural Networks (SNNs) to create ultra-low-energy game-playing agents for complex environments like **Atari Breakout**.

---

## 🌟 Objective & Core Achievement

Traditional Deep Q-Networks (DQNs) rely on continuous floating-point activations (such as ReLU) and dense matrix operations, resulting in substantial computational overhead and power consumption. Biological brains, by contrast, compute asynchronously using sparse binary spikes.

### Core Achievement
Directly training Spiking Neural Networks in sparse-reward reinforcement learning settings using surrogate gradient backpropagation often leads to severe temporal latency and training instability. To overcome this challenge, **BCS Spikeverse** pioneers an **ANN-to-SNN Weight Transfer Pipeline**:
1. **Continuous Pre-training:** A Deep Q-Network (`DQNNet`) is trained to high proficiency on `BreakoutNoFrameskip-v4` using temporal frame differencing and frame stacking.
2. **Neuromorphic Conversion:** Exact continuous weights are transferred into a deep Leaky Integrate-and-Fire (`LIFNeuron`) spiking network.
3. **Layer-by-Layer Scaling:** To counteract sparse spike firing across deeper layers, synaptic weights are systematically scaled ($10\times$ for hidden projections and $100\times$ for action projections).
4. **Super-Human Retention:** The converted SNN agent (`SNNAgent`) evaluates incoming observations via **Poisson Rate Encoding** ($T=500\text{ ms}$) and successfully retains super-human decision-making capabilities while unlocking real-time event-driven efficiency.

---

## 🛠️ Technology Stack

* **Core Framework:** PyTorch (Autograd, Neural Network Modules, Tensor Computation)
* **RL & Simulation:** OpenAI Gym (`BreakoutNoFrameskip-v4`, `CartPole-v1`)
* **Computer Vision:** OpenCV (`cv2`), NumPy (Frame stacking, cropping, binary difference thresholding)
* **Visualization & Logging:** Matplotlib, Tqdm (Multi-panel training diagnostics, reward histograms)

---

## 📁 Repository Structure

The repository has been structured according to industry-standard MLOps practices, separating production architecture, exploratory research, and formal documentation:

```text
BCS-Spikeverse/
│
├── src/                                   # Core Final Project Architecture
│   ├── config.py                          # Global hyperparameter registry (RL & LIF dynamics)
│   ├── Environment.py                     # Atari Breakout wrapper & FrameStack processing
│   ├── Neuron.py                          # LIF Spiking Neuron & adaptive threshold mechanics
│   ├── Architectures.py                   # Continuous DQNNet & multi-step temporal SNN models
│   ├── Agent.py                           # ReplayBuffer, Poisson encoder, ANNAgent & SNNAgent
│   ├── Train.py                           # Main training engine & multi-panel visualizer
│   └── Test.py                            # Evaluation benchmark & reward distribution logger
│
├── docs/                                  # Formal Project Documentation
│   └── Documentation.pdf                  # 16-page technical research paper & proof analysis
│
├── assignments/                           # Foundational Modules & Phase Research
│   ├── Harshil_spikeverse_assignment_01.ipynb # Tabular Q-Learning, GridWorld & CartPole DQN
│   ├── PER_Phase2.ipynb                   # Binary SumTree Prioritized Experience Replay (PER)
│   └── Assignment_3/                      # Direct supervised LIF SNN training on MNIST
│
├── weights/                               # Saved ANN baseline and converted SNN checkpoints
├── plots/                                 # Training trajectories and evaluation charts
├── requirements.txt                       # Project package dependencies
└── README.md                              # Project overview and usage instructions
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/BCS-Spikeverse.git
   cd BCS-Spikeverse
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

All core execution scripts reside within the `src/` directory.

### 1. Fine-Tuning & Model Training
To run the main training loop, convert pre-trained ANN weights to SNN weights, execute fine-tuning, and generate diagnostic plots:

```bash
python src/Train.py
```
* **Output:** Saves converted weights to `weights/snn_model_weights.pth` and exports a 4-panel training trajectory plot to `plots/SNN_Fine_Tune.png`.

### 2. Spiking Agent Evaluation
To run an independent benchmark evaluating the inference capability of the spiking agent over 100 test episodes:

```bash
python src/Test.py
```
* **Output:** Logs mean and maximum reward achievements and exports a reward frequency histogram to `plots/SNN_binary_test_rewards.png`.

---

## 🔬 Architectural Highlights

* **Adaptive LIF Neurons:** Features dynamic firing threshold adaptation (`theta_plus`) to stabilize firing rates during intense gameplay sequences.
* **Poisson Rate Coding:** Transforms continuous pixel intensities into stochastic binary spike trains over discrete time steps.
* **Prioritized Experience Replay:** Optimized $O(\log N)$ sampling proportional to temporal difference error using binary SumTree structures.

---

## 👥 Contributors & Acknowledgments
* **Organization:** Brain and Cognitive Science (BCS) Club, IIT Kanpur
* **Mentors:** Gaurav Rampuria, Kshitiz Tyagi, Saubhagya Pandey
