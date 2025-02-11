# 🚗 MountainCar Reinforcement Learning

This project implements **Q-learning** and **SARSA** to solve the **MountainCar-v0** environment using **OpenAI Gym**. The agent learns to drive up a hill by optimizing its policy using reinforcement learning.

## 🔹 Features
- **Q-learning & SARSA** for reinforcement learning training
- **Hyperparameter tuning via Grid Search**
- **User-friendly menu interface (`main.py`) for easy execution**
- **Modular structure for easy extension to other RL algorithms**
- **Configurable hyperparameters via `config.py`**

---

## 🛠 Installation
Ensure you have **Python 3.10+** installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project
### **Start the Interactive Menu**
Everything—training, testing, and hyperparameter tuning—can be accessed via:

```bash
python main.py
```

You'll be presented with a **menu** where you can:  
1️⃣ **Testing before learning (Random actions)**  
2️⃣ **Train using Q-learning**  
3️⃣ **Train using SARSA**  
4️⃣ **Grid Search for Q-learning**  
5️⃣ **Grid Search for SARSA**  
6️⃣ **Quit**  

Simply **enter the number** corresponding to your desired action.

---

## 🔧 Modifying Hyperparameters
The **default hyperparameters** are stored in `config.py`. You can **manually update them** before training. The current hyperparamters are the optimal hyperparamters found with grid search.

The program will automatically use the updated hyperparameters.

---

## 📊 Running Grid Search for Best Hyperparameters
To find the optimal hyperparameters using **grid search**, run grid search from the menu.

- The best hyperparameters will be printed and stored into a json file.
- You can manually update `config.py` with these values.
- Grid search **does NOT automatically modify `config.py`**.

---
