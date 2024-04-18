
Welcome to the Connect4 AI project, where advanced Reinforcement Learning techniques meet strategic gameplay. This project involves developing an AI to play the classic game Connect4, using state-of-the-art reinforcement learning methods including Distributional Dueling Networks, Noisy Nets, and Prioritized Experience Replay, all implemented using PyTorch.

## Project Overview

This AI was designed not only to understand the rules of Connect4 but to master it by competing against various strategies. It is integrated into a custom game environment and connected to a real-time interface via a Flask API. The entire setup is optimized for CUDA-enabled GPUs to enhance training efficiency and model evaluation.

## Features

- **Advanced Reinforcement Learning Models:** Utilizes Distributional Dueling Networks, Noisy Nets, and Prioritized Experience Replay to effectively learn and adapt strategies.
- **Custom Game Environment:** A tailored environment that closely simulates real-world gameplay with a digital twist.
- **Flask API Integration:** Allows for real-time interaction with the AI through a web-based interface.
- **CUDA Optimization:** Leverages GPU acceleration for efficient training and real-time data processing.
- **TensorBoard Integration:** Monitors performance metrics and visualizes training progress to assist in the neural network's fine-tuning.
- **Opponent Modeling:** Enhances AI adaptability by simulating and learning from various opponent strategies.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU for optimal performance
- Flask
- PyTorch
- TensorBoard

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/connect4-ai.git
2. Navigate to the project directory:
   ```bash
   cd connect4-ai
3. Install the required packages:
   ```bash
   pip install -r requirements.txt

### Configuration
- Ensure your system is equipped with a compatible NVIDIA GPU for CUDA.
- Set up your environment to support CUDA if not already configured.

### Usage
1. Start the Flask API
   ```bash
   python app.py
2. Access the API through your browser or use a client like Postman to interact with the AI in real-time.
