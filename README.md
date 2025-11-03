# Hangman AI - Hybrid HMM + Reinforcement Learning

An intelligent Hangman assistant that combines Hidden Markov Models (HMM) with Q-Learning to achieve optimal gameplay.

## 📋 Project Overview

This project implements a sophisticated AI agent for playing Hangman by:
- Learning statistical patterns from a 50,000-word corpus using HMM
- Optimizing decision-making through Q-Learning
- Achieving high success rates while minimizing wrong and repeated guesses

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
numpy
pandas
matplotlib
seaborn
```

### Installation

1. Ensure your data files are in the correct location:
   ```
   ml pro/
   ├── Data/
   │   ├── corpus.txt    # 50,000 word corpus
   │   └── test.txt      # Optional test file
   ├── hangman_ai.ipynb  # Main notebook
   ├── analysis_report.md
   └── README.md
   ```

2. Open `hangman_ai.ipynb` in Jupyter Notebook or VS Code

3. Run all cells sequentially to:
   - Load and analyze the corpus
   - Train HMM models
   - Train the Q-Learning agent
   - Evaluate on 2,000 test games
   - Generate visualizations and analysis

## 📓 Notebook Structure

The `hangman_ai.ipynb` notebook contains:

1. **Section 1-2**: Import libraries and load corpus data
2. **Section 3**: Exploratory data analysis with visualizations
3. **Section 4-5**: HMM implementation and training
4. **Section 6**: Hangman game environment
5. **Section 7-8**: Q-Learning agent implementation
6. **Section 9**: Training loop (5,000 episodes)
7. **Section 10-11**: Evaluation on 2,000 test games and metrics
8. **Section 12**: Visualization of learning curves
9. **Section 13**: Detailed analysis and insights
10. **Section 14**: Save models and results
11. **Section 15**: Demo gameplay examples

## 🎯 Key Features

### Hidden Markov Model (HMM)
- Position-based letter frequency tracking
- Pattern matching for candidate word filtering
- Length-specific models for each word size
- Fast probability computation

### Q-Learning Agent
- Tabular Q-learning with state encoding
- Epsilon-greedy exploration with decay
- Hybrid action selection (60% HMM + 40% Q-values)
- Reward shaping for optimal learning

### Evaluation Metrics
- **Success Rate**: Percentage of games won
- **Wrong Guesses**: Average incorrect guesses per game
- **Repeated Guesses**: Average duplicate guesses per game
- **Final Score**: Success_Rate × 2000 - Wrong × 5 - Repeated × 2

## 📊 Expected Results

After training, you should achieve:
- **Success Rate**: 60-80% (varies with training)
- **Average Wrong Guesses**: 3-5 per game
- **Average Repeated Guesses**: < 0.1 per game
- **Final Score**: 800-1400+ (depending on performance)

## 🔧 Customization

### Hyperparameters (Section 9)

```python
agent = QLearningAgent(
    hmm_models=hmm_models,
    learning_rate=0.1,        # Adjust learning rate
    discount_factor=0.95,     # Adjust discount factor
    epsilon=1.0,              # Initial exploration rate
    epsilon_decay=0.995,      # Exploration decay
    epsilon_min=0.01          # Minimum exploration
)
```

### Training Episodes

```python
# Increase/decrease training episodes
training_metrics = train_agent(
    agent, 
    train_words, 
    num_episodes=5000,  # Adjust this
    print_every=500
)
```

### Evaluation Games

```python
# Change number of test games
eval_results = evaluate_agent(
    agent, 
    test_words, 
    num_games=2000  # Adjust this
)
```

## 📈 Outputs

After running the notebook, you'll get:

### Files Generated
- `trained_agent.pkl`: Saved Q-Learning agent
- `evaluation_results.pkl`: Performance metrics
- `game_results.csv`: Detailed game-by-game results
- `training_results.png`: Learning curve visualizations

### Visualizations
- Training reward progression
- Success rate over time
- Wrong/repeated guesses trends
- Epsilon decay curve
- Performance by word length

## 📖 Documentation

See `analysis_report.md` for detailed documentation including:
- HMM design and implementation details
- RL agent architecture and training process
- Performance analysis and insights
- Challenges and solutions
- Future improvement suggestions

## 🎮 Demo Games

The notebook includes a demo section (Section 15) that plays sample games with detailed step-by-step output showing:
- Current masked word
- Available letters
- HMM probability suggestions
- Agent's chosen guess
- Outcome and rewards

## 🔬 Experimentation Ideas

1. **Try different HMM weights**:
   ```python
   # In choose_action method
   scores[action] = 0.7 * hmm_prob + 0.3 * q_value  # Instead of 0.6/0.4
   ```

2. **Adjust reward structure**:
   ```python
   # In HangmanEnvironment.step()
   reward = 2  # Increase correct guess reward
   reward = -2  # Increase wrong guess penalty
   ```

3. **Test different exploration strategies**:
   - Linear epsilon decay
   - Exponential decay
   - Boltzmann exploration

4. **Add new features to state representation**:
   - Vowel count
   - Consonant count
   - Common prefix/suffix indicators

## 🐛 Troubleshooting

### Common Issues

1. **FileNotFoundError: corpus.txt**
   - Ensure `Data/corpus.txt` exists
   - Check file path is relative to notebook location

2. **Memory Error during training**
   - Reduce `num_episodes` in training
   - Clear Q-table periodically for very long training

3. **Slow training**
   - Training 5,000 episodes takes 10-30 minutes (normal)
   - Reduce episodes or use smaller corpus for testing

4. **Poor performance**
   - Ensure sufficient training episodes (3,000+)
   - Check epsilon decay parameters
   - Verify corpus loaded correctly

## 📚 Learning Resources

- **Reinforcement Learning**: Sutton & Barto's "Reinforcement Learning: An Introduction"
- **Hidden Markov Models**: Rabiner's HMM tutorial
- **Q-Learning**: Watkins & Dayan (1992)

## 🤝 Contributing

This is a course project. For improvements:
1. Experiment with the hyperparameters
2. Try advanced RL algorithms (DQN, PPO)
3. Enhance the HMM with n-gram models
4. Add visualization improvements

## 📝 Citation

If you use this code, please cite:
```
Hangman AI - Hybrid HMM + Q-Learning Approach
Course Project, 2025
```

## 📄 License

This project is for educational purposes.

---

**Happy Training! 🎉**

For questions or issues, refer to the detailed `analysis_report.md` or review the inline comments in the notebook.
