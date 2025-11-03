# Hangman AI - Quick Setup and Execution Guide

## ✅ Project Setup Complete!

Your project structure is now:

```
ml pro/
├── hangman_ai.ipynb          # Main implementation notebook (READY TO RUN!)
├── analysis_report.md         # Comprehensive analysis and documentation
├── README.md                  # Quick start guide
├── SETUP_GUIDE.md            # This file
└── Data/
    ├── corpus.txt            # 50,000 word corpus ✓
    └── test.txt              # Test file ✓
```

## 🚀 How to Run the Project

### Step 1: Install Required Packages

If you haven't already, install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn
```

Or if using conda:

```bash
conda install numpy pandas matplotlib seaborn
```

### Step 2: Open the Notebook

1. **In VS Code** (Recommended):
   - Open `hangman_ai.ipynb`
   - Select a Python kernel (3.8+)
   - Run cells sequentially using Shift+Enter

2. **In Jupyter Notebook**:
   ```bash
   jupyter notebook hangman_ai.ipynb
   ```

3. **In Jupyter Lab**:
   ```bash
   jupyter lab hangman_ai.ipynb
   ```

### Step 3: Run the Notebook

**Option A: Run All Cells**
- In VS Code: Click "Run All" at the top of the notebook
- In Jupyter: Menu → Cell → Run All

**Option B: Run Step-by-Step** (Recommended for first time)
1. Run cells 1-2: Import libraries and load data (~5 seconds)
2. Run cell 3: View data analysis and plots (~10 seconds)
3. Run cells 4-5: Train HMM models (~10 seconds)
4. Run cells 6-8: Set up environment and agent (~1 second)
5. Run cell 9-10: Train agent (~10-30 minutes) ⏰
6. Run cell 11: Evaluate on 2000 games (~5-10 minutes) ⏰
7. Run cells 12-15: Generate visualizations and analysis (~30 seconds)

### Step 4: Review Results

After running all cells, you'll have:
- Training metrics and learning curves
- Evaluation results and final score
- Performance breakdown by word length
- Saved model files (`.pkl`)
- Visualization images (`.png`)

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Data loading & EDA | 15 seconds |
| HMM training | 10 seconds |
| RL agent training (5000 episodes) | 10-30 minutes |
| Evaluation (2000 games) | 5-10 minutes |
| Visualization & analysis | 30 seconds |
| **Total** | **15-45 minutes** |

## 🎯 Expected Outputs

### Console Output

You should see progress messages like:

```
Libraries imported successfully!
Loading corpus...
Total words loaded: 50000
Training HMM models...
  Trained HMM for length 5 with 1234 words
  ...
Starting training for 5000 episodes...
Episode 500/5000
  Win Rate: 45.2%
  Avg Reward: 2.34
  ...
Evaluating agent on 2000 games...
FINAL SCORE: 1245.00
```

### Generated Files

- `trained_agent.pkl` - Your trained Q-Learning agent
- `evaluation_results.pkl` - Performance metrics
- `game_results.csv` - Detailed results for each test game
- `training_results.png` - Comprehensive visualization

### Visualizations

You'll see 6 plots showing:
1. Training rewards over time
2. Success rate progression
3. Wrong guesses trend
4. Repeated guesses trend
5. Epsilon decay curve
6. Success rate by word length

## 🔧 Customization Tips

### Quick Experiments

**Faster Training (for testing):**
```python
# In Section 9, change:
training_metrics = train_agent(agent, train_words, num_episodes=1000)  # Instead of 5000
```

**Faster Evaluation (for testing):**
```python
# In Section 10, change:
eval_results = evaluate_agent(agent, test_words, num_games=500)  # Instead of 2000
```

**Different HMM/RL Balance:**
```python
# In QLearningAgent.choose_action(), change:
scores[action] = 0.7 * hmm_prob + 0.3 * (q_value + 1) / 2  # More HMM weight
# or
scores[action] = 0.5 * hmm_prob + 0.5 * (q_value + 1) / 2  # Equal weight
```

## 📊 Understanding Your Results

### Good Performance Indicators

✅ Success rate > 60%
✅ Average wrong guesses < 5
✅ Repeated guesses < 0.1
✅ Final score > 900

### Interpretation

**Final Score Formula:**
```
Final Score = Success_Rate × 2000 - Wrong_Guesses × 5 - Repeated_Guesses × 2
```

**Example:**
- 70% success rate (1400 points)
- 800 wrong guesses (-4000 points)
- 10 repeated guesses (-20 points)
- **Final Score: 1400 - 4000 - 20 = -2620** ❌ Too many wrong guesses!

**Better Example:**
- 70% success rate (1400 points)
- 400 wrong guesses (-2000 points)
- 5 repeated guesses (-10 points)
- **Final Score: 1400 - 2000 - 10 = -610** ❌ Still need improvement!

**Good Example:**
- 75% success rate (1500 points)
- 200 wrong guesses (-1000 points)
- 2 repeated guesses (-4 points)
- **Final Score: 1500 - 1000 - 4 = 496** ✅ Good!

## 🐛 Common Issues & Solutions

### Issue 1: "No module named 'numpy'"
**Solution:**
```bash
pip install numpy pandas matplotlib seaborn
```

### Issue 2: "FileNotFoundError: Data/corpus.txt"
**Solution:**
- Check that you're running from the `ml pro` directory
- Verify the file path in cell 2 matches your structure

### Issue 3: Training takes too long
**Solutions:**
- Reduce `num_episodes` to 1000 for quick testing
- Ensure no other heavy processes are running
- Normal training time is 10-30 minutes

### Issue 4: Kernel dies during training
**Solutions:**
- Reduce `num_episodes` to prevent memory issues
- Restart kernel and run again
- Close other applications to free memory

### Issue 5: Low performance (< 50% success rate)
**Solutions:**
- Ensure training completed all episodes
- Check epsilon decay parameters
- Verify corpus loaded correctly (should be ~50,000 words)
- Try training for more episodes (7000-10000)

## 📈 Performance Optimization

### To Improve Success Rate:

1. **Train Longer**:
   ```python
   training_metrics = train_agent(agent, train_words, num_episodes=10000)
   ```

2. **Adjust Learning Rate**:
   ```python
   agent = QLearningAgent(..., learning_rate=0.15)  # Higher for faster learning
   ```

3. **Slower Epsilon Decay**:
   ```python
   agent = QLearningAgent(..., epsilon_decay=0.997)  # Explore longer
   ```

4. **Higher HMM Weight**:
   ```python
   scores[action] = 0.7 * hmm_prob + 0.3 * normalized_q  # Trust corpus more
   ```

## 📚 Next Steps After Running

1. **Review the Results** (Section 11)
   - Check your final score
   - Analyze performance by word length
   - Identify patterns in failures

2. **Study the Visualizations** (Section 12)
   - Observe learning curves
   - Check convergence
   - Verify epsilon decay

3. **Read the Analysis** (Section 13)
   - Understand design choices
   - Learn from challenges
   - Consider improvements

4. **Read `analysis_report.md`**
   - Deep dive into methodology
   - Understand HMM and RL details
   - Explore future directions

5. **Experiment**
   - Try different hyperparameters
   - Test alternative reward structures
   - Implement suggested improvements

## 🎓 Learning Objectives

By completing this project, you will:

✅ Understand Hidden Markov Models for sequence prediction
✅ Implement Q-Learning from scratch
✅ Combine statistical models with reinforcement learning
✅ Handle state/action representation in RL
✅ Implement exploration-exploitation strategies
✅ Evaluate and analyze ML model performance
✅ Visualize learning curves and metrics

## 💡 Tips for Success

1. **Run incrementally**: Don't run all cells at once on first try
2. **Monitor training**: Watch the progress messages to ensure proper training
3. **Save checkpoints**: The notebook saves the trained agent automatically
4. **Experiment**: Try different parameters after initial run
5. **Document findings**: Note what works and what doesn't

## 📞 Need Help?

If you encounter issues:

1. Check the **Common Issues** section above
2. Review inline comments in the notebook
3. Read the detailed `analysis_report.md`
4. Verify all packages are installed correctly
5. Ensure Python version is 3.8 or higher

## 🎉 Ready to Go!

You're all set! Open `hangman_ai.ipynb` and start running the cells.

**Recommended Order:**
1. Read the README.md (overview)
2. Run hangman_ai.ipynb (implementation)
3. Review analysis_report.md (deep dive)

**Have fun and happy learning! 🚀**

---

*Last updated: November 3, 2025*
