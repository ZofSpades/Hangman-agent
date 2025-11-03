Goal:
Develop an intelligent Hangman assistant that learns to guess hidden words with minimal mistakes using a hybrid system: a Hidden Markov Model (HMM) for probabilistic intuition and a Reinforcement Learning (RL) agent for optimal decision-making. The solution must train and evaluate on the provided corpus.txt (50,000 words), and meet all competition requirements as stated in the problem statement.

Functionality Breakdown
1. Data Preparation

Load and preprocess the corpus.txt file, creating lists by word length and possible alphabet.

Ensure the code works only with the provided corpus, without any extra datasets.

2. Hidden Markov Model (HMM)

Implement and train an HMM on words grouped by their lengths.

Define suitable hidden states and emission probabilities so that, given a partially filled word (masked by underscores), the HMM predicts the probabilities for each letter in each unguessed position.

Allow the HMM to output a probability distribution over all possible next letters for a given masked pattern and prior guesses.

3. Hangman Game Environment

Construct an environment class that simulates Hangman with the following features:

Allows N wrong guesses (configurable, default=6).

Tracks masked word (e.g., 'a__l_'), guessed letters, remaining lives, and remaining possible words.

Accepts letter guesses and returns reward signals (see below).

4. RL Agent (Q-learning or DQN)

Define the agent's state using:

The current masked pattern.

The set of letters guessed so far.

The remaining lives.

The letter probability vector from the HMM.

Implement the agent to learn optimal guesses:

State: string and vector encoding.

Actions: pick any unguessed letter.

Reward: +1 for correct guess, -1 for incorrect guess, big penalty for repeats.

Train using Q-learning (for simple representation) or DQN (for complex vector encoding).

Use -greedy exploration with decay and/or other suitable exploration-exploitation strategies.

5. Training & Evaluation

Train the RL agent by playing simulated Hangman games using words from the training corpus, updating the policy after each episode.

For evaluation:

Play 2,000 games on a held-out test set (from corpus).

Track: Success Rate (percentage of solved words), total wrong guesses, and repeated guesses.

Calculate Final Score as described:

Final Score
=
Success Rate
×
2000
−
Wrong Guesses
×
5
−
Repeated Guesses
×
2
Final Score=Success Rate×2000−Wrong Guesses×5−Repeated Guesses×2

6. Visualization & Analysis

Plot learning curves: reward per episode, changes in success rate, wrong/repeated guesses over time.

Generate a brief summary (.pdf or .md) analyzing:

HMM design and impact.

RL agent structure and exploration strategy.

Difficulties encountered and future improvements.

Technical Notes
Organize code into clear sections/notebooks:

Data/HMM training.

RL environment and training loop.

Evaluation and plotting.

Report code/scripts.

Write clean, modular functions and classes.

Add docstrings and comments everywhere for automated code assistance.

Do not use external pre-trained models or wordlists, only the provided corpus.

Input

A plain text corpus.txt (provided, 50,000 English words, each on a new line).

Output

Notebook/scripts with:

HMM construction and training.

RL agent and Hangman environment implementation.

Training, evaluation, and plotting.

Summary of results and insights.

Well-structured, commented code suitable for step-by-step Copilot completion.

Deliverables

Python scripts or Jupyter notebooks covering all code and analysis steps above.

A separate analysis report in PDF answering:

Key challenges and insights.

HMM and RL design choices.

Exploration/exploitation techniques.

Ideas for future improvement.

Start the implementation by building a data loader and HMM trainer for the Hangman corpus, then create the Hangman environment class, and finally code the RL agent with all associated training and evaluation routines.

