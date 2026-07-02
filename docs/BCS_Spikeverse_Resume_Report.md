# BCS Spikeverse: Exhaustive Project Dossier & Resume-Engineering Guide

> **Purpose of this Document**  
> This comprehensive dossier is designed to serve as an **all-inclusive context document and prompt injection** for AI agents (LLMs) or recruiters crafting elite, highly quantifiable resume bullet points. It details every mathematical mechanism, architectural decision, empirical benchmark, and technical innovation within the **BCS Spikeverse** project repository.

---

## 1. Project Overview & Institutional Context

* **Project Title:** BCS Spikeverse — Reinforcement Learning with Spiking Neural Networks
* **Domain:** Deep Reinforcement Learning (RL), Neuromorphic Computing, Spiking Neural Networks (SNNs), Computer Vision.
* **Affiliation:** Brain and Cognitive Science (BCS) Club, Indian Institute of Technology (IIT) Kanpur.
* **Core Problem Statement:** Modern Deep Q-Networks (DQNs) achieve superhuman performance in complex environments (e.g., Atari games) but rely on dense, continuous floating-point matrix multiplications (Von Neumann architecture bottleneck), resulting in massive energy consumption. Spiking Neural Networks (SNNs) process asynchronous binary events (spikes), offering orders-of-magnitude energy savings on neuromorphic hardware (e.g., Intel Loihi, IBM TrueNorth). However, directly training deep SNNs in sparse-reward RL environments using surrogate gradient backpropagation suffers from extreme temporal latency and training instability.
* **Primary Contribution:** Developed an **ANN-to-SNN Weight Transfer & Layer-Scaling Pipeline**. Trained continuous Convolutional/Dense Deep Q-Networks (`DQNNet`) to mastery on `BreakoutNoFrameskip-v4` and transferred exact weights into a deep Leaky Integrate-and-Fire (`LIFNeuron`) spiking network. By introducing layer-wise synaptic scaling ($10\times$ and $100\times$) and **Poisson Rate Encoding** ($T=500\text{ ms}$), the SNN retained superhuman game-playing capability while eliminating direct training instability.

---

## 2. Technical Architecture & Algorithmic Mechanics

### 2.1 Spiking Neuron Dynamics (`src/Neuron.py`)
* **Leaky Integrate-and-Fire (LIF) Model:** Membrane potential $V(t)$ integrates synaptic input currents $I(t)$ and decays exponentially toward resting potential $V_{rest} = -65\text{ mV}$ with decay factor $\alpha = 0.01$:
  $$V(t) = V(t-1) + \left(-V(t-1) + V_{rest}\right)\cdot \alpha + I(t)$$
* **Adaptive Firing Threshold ($\theta$):** To prevent hyper-excitation and bursting during high-intensity game sequences, neurons incorporate dynamic threshold adaptation. When membrane voltage crosses $V_{thresh} = -52\text{ mV} + \theta(t)$, the neuron emits a binary spike $S \in \{0, 1\}$, resets membrane potential to $0\text{ V}$, and adapts the threshold:
  $$\theta(t+1) = \theta(t) + \theta_{plus}\cdot S - \theta_{decay}\cdot \theta(t) \quad (\theta_{plus}=0.05, \theta_{decay}=10^{-7})$$
* **Surrogate Gradient Backpropagation (`SurrogateSpike`):** For direct supervised training experiments (MNIST), the non-differentiable Heaviside step function is replaced during the backward pass with a rational exponential surrogate derivative ($\beta=10.0$):
  $$\frac{\partial S}{\partial V} = \beta \frac{e^{-\beta|V - V_{th}|}}{\left(1 + e^{-\beta|V - V_{th}|}\right)^2}$$

### 2.2 Deep Q-Network & Advanced RL Pipeline (`src/Architectures.py`, `src/Agent.py`)
* **Continuous Baseline (`DQNNet`):** 2-layer feedforward architecture (`6400 -> 1000 -> ReLU -> 4 actions`) mapping 80x80 flattened processed Atari frames to discrete joystick actions.
* **Temporal SNN Architecture (`SNN`):** 2-layer spiking architecture (`SNNLayer`) simulating temporal evolution across $T=500\text{ ms}$ discrete time steps. Integrates input spikes over time and executes the action corresponding to the maximum accumulated output spike count.
* **Poisson Rate Encoding (`poisson_spike_encoding`):** Converts continuous normalized pixel intensities $I \in [0, 1]$ into 3D binary spike trains `[batch, time_steps=500, features=6400]` by sampling against uniform random noise generated at each timestep.
* **Prioritized Experience Replay (`assignments/PER_Phase2.ipynb`):** Implemented a custom binary **SumTree** data structure ($2N-1$ nodes) allowing $O(\log N)$ priority updates and stratified sampling. Experiences are sampled proportional to absolute Temporal Difference (TD) error $|\delta_i|^\alpha$ ($\alpha=0.6$), with importance sampling bias correction weights $w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$ annealed from $\beta_0=0.4 \to 1.0$.

### 2.3 Computer Vision & Temporal Frame Processing (`src/Environment.py`)
* **Atari Frame Stacking (`FrameStack`):** Maintains a circular buffer (`deque`) of 4 consecutive frames (`agent_history_length=4`) with frame-skipping (`frame_skip=3`, `action_repeat=4`).
* **Scoreboard Cropping:** Crops the top 10% scoreboard area (`img[10*h//100:, :]`) to isolate the active 50% gameplay screen.
* **Binary Difference Processing (`get_binary`):** Computes frame-to-frame motion differences (`curr - prev`), thresholding positive velocity shifts into an 80x80 binary representation capturing ball trajectory and paddle speed.
* **Weighted Greyscale Fading (`get_greyscale`):** Applies exponential historical blending weights `[1.0, 0.75, 0.5, 0.25]` across stacked difference frames to preserve fading motion trails.

---

## 3. Comprehensive Quantitative Benchmarks & Metrics Registry

This table aggregates every exact number across the repository for immediate retrieval by downstream resume-generation agents:

| Category | Metric / Parameter | Exact Value | Significance / Context |
| :--- | :--- | :--- | :--- |
| **RL Training** | Total Training Episodes | `4,000` episodes | Pre-training duration on Atari Breakout |
| **RL Memory** | Replay Buffer Capacity | `200,000` transitions | Circular replay buffer (`init_size = 50,000`) |
| **Optimization** | Optimizer & Learning Rate | `RMSprop`, `lr = 1e-5` | Gradient momentum = `0.95`, mini-batch size = `64` |
| **Exploration** | $\epsilon$-Greedy Schedule | `1.0` $\to$ `0.1` | Linear annealing over `200,000` exploration steps |
| **SNN Dynamics** | Simulation Time Window ($T$) | `500` timesteps | Millisecond temporal window for rate-coded spiking |
| **Weight Transfer**| Synaptic Scaling Factors | Layer 1: `10x`, Layer 2: `100x` | Compensates for spike sparsity in deeper layers |
| **Atari Performance**| Baseline ANN Average Score | **`169.0`** points | Rolling 50-episode mean (Human baseline $\approx 32.0$) |
| **Atari Performance**| Peak Agent Episode Score | **`290.0`** points | Superhuman gameplay mastery achieved |
| **Atari SNN** | Converted SNN Evaluation | Retained super-human mastery | Maintained policy without backprop latency |
| **Atari Reward** | Mean Reward Frequency | **`1.13`** (SNN) vs `0.63` (ANN) | Demonstrated improved reward consistency per evaluation |
| **Vision Benchmark**| Direct SNN on MNIST | **`94.3%`** accuracy in 3 epochs | Assignment 3 direct surrogate gradient training |
| **Vision Benchmark**| Converted SNN on MNIST | **`96.84%`** accuracy in 10 epochs | Converted via Poisson rate encoding |
| **Computational** | Direct SNN Training Speed | $\approx 15$ mins / 50 episodes | Highlighted temporal latency motivating conversion |

---

## 4. Drafted 4–5 Bullet Point Frameworks (Resume Ready)

Below are three specialized options (4–5 bullets each) ready to be copied directly into a resume or refined by an AI agent depending on the target role.

### Option A: Deep Reinforcement Learning & AI Engineer Focus
* Engineered an end-to-end Deep Reinforcement Learning framework in PyTorch interfacing with OpenAI Gym (`BreakoutNoFrameskip-v4`), utilizing multi-frame temporal difference preprocessing and frame-skipping to process 80x80 game states.
* Implemented a binary **SumTree Prioritized Experience Replay (PER)** memory buffer ($O(\log N)$ updates), sampling 200,000 temporal transitions proportional to TD error $|\delta_i|^{0.6}$ with annealed importance sampling weights ($\beta=0.4 \to 1.0$).
* Developed an **ANN-to-SNN Weight Transfer Pipeline**, converting continuous Deep Q-Networks into event-driven Spiking Neural Networks via layer-wise synaptic scaling ($10\times$ and $100\times$) to bypass direct surrogate gradient training instability.
* Simulated biological Leaky Integrate-and-Fire (`LIFNeuron`) dynamics over 500ms temporal windows using **Poisson Rate Encoding**, incorporating adaptive firing thresholds ($\theta_{plus}=0.05$) to stabilize spiking patterns during high-frequency gameplay.
* Achieved superhuman Atari gameplay mastery with a peak episode score of **290.0** (rolling mean **169.0** vs. 32.0 human baseline), retaining full decision policy in the converted SNN while demonstrating an improved reward distribution mean (**1.13** vs. `0.63`).

### Option B: Research Engineer / Neuromorphic Computing Focus
* Spearheaded research on Neuromorphic Reinforcement Learning within the Brain and Cognitive Science Club (IIT Kanpur), evaluating computational trade-offs between 2nd-gen continuous ANNs and 3rd-gen event-driven Spiking Neural Networks (SNNs).
* Architected custom PyTorch autograd layers (`SurrogateSpike`) implementing rational exponential surrogate derivatives ($\beta=10.0$) to enable backpropagation through discrete binary membrane threshold crossings.
* Demonstrated that direct supervised SNN training achieves **94.3%** MNIST classification in 3 epochs, while identifying extreme temporal latency ($\sim 15$ mins/50 episodes) when applying direct STDP/surrogates to deep sparse-reward RL tasks.
* Overcame sparse RL training divergence by engineering a layer-by-layer weight scaling transfer methodology ($10\times/100\times$), achieving **96.84%** converted accuracy on MNIST and preserving superhuman decision boundaries in Atari Breakout.
* Formulated adaptive LIF neuron membrane differential equations ($V_{rest}=-65\text{mV}$, decay $=0.01$) integrated with 500-step Poisson rate encoders, laying the algorithmic groundwork for ultra-low-power edge deployment on neuromorphic chips.

---

## 5. System Prompt & Context Injection for Downstream AI Agents

If you are feeding this dossier into another LLM (e.g., ChatGPT, Claude, Gemini) to custom-tailor your resume points, **copy and paste the block below directly into the prompt**:

```text
[SYSTEM INSTRUCTIONS FOR RESUME AGENT]
You are an expert technical resume writer and senior MLOps/AI engineering hiring manager. I am providing you with the complete technical dossier of my project, "BCS Spikeverse — Reinforcement Learning with Spiking Neural Networks" (IIT Kanpur).

Your task: Craft exactly 4 to 5 elite, resume-ready bullet points based strictly on the metrics and architectures provided in the dossier above.

Follow these strict writing rules:
1. Start every bullet point with a strong, active engineering verb (e.g., Engineered, Architected, Implemented, Spearheaded, Formulated, Developed).
2. Quantify results rigorously using the exact metrics from the dossier (e.g., peak score 290.0, 200k replay buffer, O(log N) SumTree PER, 10x/100x weight scaling, 500ms Poisson encoding, 96.84% MNIST accuracy).
3. Emphasize the "Problem -> Approach -> Impact" structure: explain *why* ANN-to-SNN weight transfer was needed (overcoming direct surrogate gradient training instability/latency) and *what* it achieved (retaining superhuman Atari Breakout mastery with neuromorphic energy efficiency).
4. Integrate industry-standard keywords naturally: PyTorch, OpenAI Gym, Deep Q-Networks (DQN), Prioritized Experience Replay (SumTree), Spiking Neural Networks (SNN), Leaky Integrate-and-Fire (LIF), Poisson Rate Encoding, Computer Vision (OpenCV).
```
