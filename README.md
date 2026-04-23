🚀 Flappy Bird AI using NEAT & Pygame

This project is a recreation of the classic Flappy Bird game using Python and Pygame, enhanced with an AI that learns to play automatically using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

Instead of manually controlling the bird, a population of 50 AI-controlled birds is generated. Each bird is powered by its own neural network, which decides whether to jump based on its environment. Over time, the AI improves through evolution — birds that survive longer receive higher fitness scores and are more likely to pass their traits to the next generation.

🧠 How it works

The neural network takes inputs such as:

Bird’s vertical position
Distance from the top pipe
Distance from the bottom pipe

And outputs:

Jump or do nothing

Through multiple generations, the AI gradually learns to navigate the pipes efficiently.

🎮 Features
Built with Pygame
AI powered by NEAT-Python
Real-time evolution and learning
Score tracking system
Custom PNG graphics (bird, pipes, background)
▶️ Run the project
pip install pygame neat-python
python main.py

This project is great for understanding AI, neuroevolution, and game development in a simple and visual way.
