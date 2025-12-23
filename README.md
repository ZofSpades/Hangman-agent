# Intelligent Hangman AI Assistant

A sophisticated Hangman AI that combines **Hidden Markov Models (HMM)** with **Reinforcement Learning (Q-Learning)** to achieve optimal letter prediction and game-playing strategies.

## 🎯 Project Overview

This project implements an intelligent agent that plays Hangman by learning from a corpus of words and making strategic letter guesses. The hybrid approach leverages statistical patterns from HMM and adaptive decision-making from Q-Learning to maximize success rates.

## ✨ Key Features

### Advanced HMM Implementation
- **Trigram Analysis**: 3-letter sequence prediction for better context
- **Position-Specific Bigrams**: Tracks starting/ending letter pairs
- **Common Ending Detection**: Recognizes patterns like -ing, -ed, -tion, -ly, -ness, -ment, etc.
- **Vowel/Consonant Balancing**: Smart detection and prioritization
- **Adaptive Weighting**: 6 different strategies combined dynamically based on game state

### Q-Learning Integration
- **Hybrid Training**: 80% HMM + 20% Q-Learning during training
- **Pure HMM Evaluation**: 100% HMM for maximum generalization on unseen words
- **Epsilon-Greedy Exploration**: Balanced exploration vs exploitation
- **State Encoding**: Efficient representation of game states for Q-table

## 📊 Performance Results

### Test Set Performance (100% Unseen Words from test.txt)
- **Success Rate**: 30.60% (612/2000 games won)
- **Average Wrong Guesses**: 5.25 per game
- **Repeated Guesses**: 0 (Perfect letter tracking)
- **Final Score**: -51,913

### Training Performance (corpus.txt)
- **Training Win Rate**: 95-96%
- **Training Episodes**: 5,000
- **Convergence**: ~500 episodes

### Performance by Word Length
| Word Length | Success Rate | Performance |
|-------------|--------------|-------------|
| 2-7 letters | 0-21% | Challenging |
| 8-14 letters | 23-45% | Moderate |
| 15-19 letters | 61-75% | Strong |
| 20-22 letters | 100% | Excellent |

**Best Performance**: 100% success on 20-letter and 22-letter words

## 🚀 Installation

### Clone the Repository
```bash
git clone https://github.com/ZofSpades/Hangman-agent.git
cd Hangman-agent
```

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn
```

## 📁 Project Structure

```
Hangman-agent/
│
├── hangman_ai.ipynb          # Main Jupyter notebook with full implementation
├── README.md                  # Project documentation
├── prompt.md                  # Project prompt/requirements
├── Problem_Statement.pdf      # Original problem statement
│
└── Data/
    ├── corpus.txt            # Training data (49,979 words)
    └── test.txt              # Test data (2,000 unseen words)
```

## 🎮 Usage

### Running the Complete Pipeline

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook hangman_ai.ipynb
   ```

2. **Execute All Cells**: Run cells sequentially from top to bottom
   - Cells 1-2: Import libraries and load data
   - Cell 3: Visualize data distribution
   - Cells 4-5: Train HMM models
   - Cells 6-8: Define game environment and RL agent
   - Cell 9: Train the Q-Learning agent (5000 episodes)
   - Cells 10-11: Evaluate on test set and view results

### Quick Start Example

```python
# Load and train HMM
from hangman_ai import HangmanHMM, QLearningAgent

# Load corpus
train_words = load_corpus('Data/corpus.txt')

# Train HMM models
hmm_models = {}
for length, words in organize_by_length(train_words).items():
    hmm = HangmanHMM()
    hmm.train(words)
    hmm_models[length] = hmm

# Initialize and train Q-Learning agent
agent = QLearningAgent(hmm_models=hmm_models)
training_metrics = train_agent(agent, train_words, num_episodes=5000)

# Evaluate on test set
test_words = load_corpus('Data/test.txt')
eval_results = evaluate_agent(agent, test_words, num_games=2000)

print(f"Success Rate: {eval_results['success_rate']*100:.2f}%")
```

## 🧠 Technical Details

### HMM Strategies (6 Total)

1. **Overall Corpus Frequency** (Weight: 0.80 first guess, varies later)
   - Tracks letter frequency across entire corpus
   - Most common: e (10.37%), a (8.87%), i (8.86%)

2. **Pattern Matching** (Weight: 0.0-0.45 based on game state)
   - Finds words matching revealed letters
   - Highest weight in late game when pattern is clear

3. **Position-Based Frequency** (Weight: 0.0-0.25)
   - Tracks letter frequency at specific positions
   - Important in early-mid game

4. **Bigram Analysis** (Weight: 0.10-0.27)
   - Analyzes letter pair frequencies
   - Predicts likely next/previous letters

5. **Trigram Analysis** (Weight: 0.0-0.22)
   - 3-letter sequence prediction
   - Strong in mid-late game

6. **Common Ending Detection** (Weight: 0.05-0.15)
   - Recognizes 15 common suffixes
   - Boosts accuracy on word endings

### Adaptive Weighting System

The system dynamically adjusts strategy weights based on:
- **First Guess**: 80% frequency + 20% vowel boost
- **Early Game (0-25% revealed)**: Focus on frequency + position
- **Mid Game (25-50% revealed)**: Introduce patterns and trigrams
- **Late Game (50%+ revealed)**: Heavy on patterns, trigrams, and endings

### Training Strategy

- **Training**: Hybrid approach (80% HMM + 20% Q-Learning)
  - Q-Learning memorizes training words
  - Achieves 95-96% success on training set

- **Evaluation**: Pure HMM (100%)
  - Uses corpus statistics for generalization
  - Better performance on completely unseen words
  - Avoids overfitting to training data

## 📈 Dataset Statistics

### Training Set (corpus.txt)
- **Total Words**: 49,979
- **Word Length Range**: 1-24 letters
- **Most Common Lengths**: 7-10 letters (peak at ~6,800 words of length 9)

### Test Set (test.txt)
- **Total Words**: 2,000
- **Completely Unseen**: No overlap with training set
- **Test/Train Ratio**: 4.0%

### Letter Frequency (Top 10)
1. e: 10.37%
2. a: 8.87%
3. i: 8.86%
4. o: 7.54%
5. r: 7.07%
6. n: 7.02%
7. t: 6.78%
8. s: 6.12%
9. l: 5.77%
10. c: 4.58%

## 🔍 Key Insights

1. **No File Creation**: The notebook runs entirely in-memory with no external file generation (no images, configs, or pickle files saved)

2. **Strict Data Separation**: 
   - Training uses 100% of corpus.txt
   - Testing uses 100% of test.txt
   - Zero mixing between datasets

3. **Perfect Letter Tracking**: Zero repeated guesses across all 2,000 test games

4. **Generalization Challenge**: The 30.6% test success rate (vs 95%+ training) indicates the test words are significantly different from training distribution

5. **Length Advantage**: Performance improves dramatically on longer words (15+ letters), suggesting more context helps the HMM strategies

## 🛠️ Hyperparameters

### Q-Learning Agent
- **Learning Rate**: 0.1
- **Discount Factor**: 0.95
- **Initial Epsilon**: 1.0
- **Epsilon Decay**: 0.995
- **Minimum Epsilon**: 0.01
- **HMM Weight**: 0.8 (during training)

### Training Configuration
- **Episodes**: 5,000
- **Max Lives per Game**: 6
- **Print Frequency**: Every 500 episodes

## 📝 Scoring Formula

```
Final Score = (Success Rate × 2000) - (Wrong Guesses × 5) - (Repeated Guesses × 2)
```

## 🤝 Contributing

This is an educational project demonstrating the integration of statistical models (HMM) with reinforcement learning (Q-Learning) for game-playing AI.

## 📄 License

This project is open source and available for educational purposes.

## 🎓 Learning Outcomes

- Implementation of Hidden Markov Models for pattern recognition
- Q-Learning for sequential decision-making
- Hybrid AI approaches combining statistical and learning-based methods
- Handling unseen data and generalization challenges
- Adaptive strategy selection based on game state

---

**Note**: The relatively modest test success rate (30.6%) reflects the challenging nature of generalizing to completely unseen words with only 6 lives per game. The model demonstrates strong performance on training data (95-96%) and excels on longer test words (60-100% on 15+ letters), showing that the HMM strategies effectively leverage contextual information when available.
