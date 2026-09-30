# Q-learning для задачи управления движением робота по клеточному полю.
# Поле 5x5, стены, старт (0,0), цель (4,4).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Параметры ----------
GRID_SIZE = 5                  # размер поля 5x5
N_EPISODES = 1000              # число эпизодов обучения
MAX_STEPS = 100                # лимит шагов на эпизод
ALPHA = 0.1                    # скорость обучения
GAMMA = 0.9                    # коэффициент дисконтирования
EPSILON_START = 1.0            # начальная вероятность исследования
EPSILON_END = 0.01             # конечная вероятность исследования
EPSILON_DECAY = 0.995          # скорость уменьшения epsilon
WINDOW = 100                   # окно скользящего среднего

# ---------- Лабиринт ----------
# 1 = стена, 0 = свободная клетка
MAZE = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0],
])

START = (0, 0)
GOAL = (4, 4)

# ---------- Действия ----------
# 0 = вверх, 1 = вниз, 2 = влево, 3 = вправо
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['↑', '↓', '←', '→']
N_ACTIONS = len(ACTIONS)

# ---------- Вспомогательные функции ----------
def is_valid(state):
    # Проверяет, что клетка внутри поля и не является стеной
    r, c = state
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and MAZE[r, c] == 0

def step(state, action):
    # Выполняет действие. Возвращает (новое_состояние, награда, конец_эпизода)
    dr, dc = ACTIONS[action]
    new_state = (state[0] + dr, state[1] + dc)

    # Попытка выйти за границу или в стену
    if not is_valid(new_state):
        return state, -10, False

    # Достигли цели
    if new_state == GOAL:
        return new_state, +10, True

    # Обычный шаг
    return new_state, -1, False

# ---------- Q-таблица ----------
# Q[(row, col), action] = ожидаемая суммарная награда
Q = np.zeros((GRID_SIZE, GRID_SIZE, N_ACTIONS))

# ---------- Обучение ----------
rewards_per_episode = []
epsilon = EPSILON_START

for episode in range(N_EPISODES):
    state = START
    total_reward = 0

    for _ in range(MAX_STEPS):
        # ε-жадная стратегия
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            action = int(np.argmax(Q[state[0], state[1]]))

        new_state, reward, done = step(state, action)

        # Обновление Q по формуле
        best_next = np.max(Q[new_state[0], new_state[1]])
        Q[state[0], state[1], action] += ALPHA * (
            reward + GAMMA * best_next - Q[state[0], state[1], action]
        )

        state = new_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

# ---------- График средней награды ----------
rewards = np.array(rewards_per_episode)
moving_avg = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, label='Награда за эпизод')
plt.plot(range(WINDOW - 1, N_EPISODES), moving_avg,
         linewidth=2, label=f'Скользящее среднее ({WINDOW} эпизодов)')
plt.xlabel('Эпизод')
plt.ylabel('Суммарная награда')
plt.title('Обучение агента (Q-learning)')
plt.legend()
plt.grid(True)

# ---------- Визуализация пути обученного агента ----------
def get_optimal_path():
    """Строит путь от старта до цели по обученной Q-таблице."""
    state = START
    path = [state]
    for _ in range(MAX_STEPS):
        action = int(np.argmax(Q[state[0], state[1]]))
        new_state, _, done = step(state, action)
        if new_state == state:  # агент упёрся в стену
            break
        path.append(new_state)
        state = new_state
        if done:
            break
    return path

path = get_optimal_path()

fig, ax = plt.subplots(figsize=(6, 6))

# Клетки
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if MAZE[r, c] == 1:
            color = 'black'
        elif (r, c) == START:
            color = 'lightgreen'
        elif (r, c) == GOAL:
            color = 'gold'
        else:
            color = 'white'
        ax.add_patch(Rectangle((c, GRID_SIZE - 1 - r), 1, 1,
                               facecolor=color, edgecolor='gray'))

# Путь стрелками
for i in range(len(path) - 1):
    r1, c1 = path[i]
    r2, c2 = path[i + 1]
    x1, y1 = c1 + 0.5, GRID_SIZE - 1 - r1 + 0.5
    x2, y2 = c2 + 0.5, GRID_SIZE - 1 - r2 + 0.5
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Подписи
ax.text(START[1] + 0.5, GRID_SIZE - 1 - START[0] + 0.5, 'S',
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(GOAL[1] + 0.5, GRID_SIZE - 1 - GOAL[0] + 0.5, 'G',
        ha='center', va='center', fontsize=14, fontweight='bold')

ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Путь обученного агента')
plt.tight_layout()
plt.show()

# ---------- Вывод Q-таблицы ----------
print("\nОбученная Q-таблица (максимальное Q по каждому состоянию):")
print("Строки — состояния (r, c), столбцы — действия ↑ ↓ ← →")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if MAZE[r, c] == 1:
            print(f"({r},{c}) — стена")
        else:
            qs = Q[r, c]
            best = ACTION_NAMES[int(np.argmax(qs))]
            print(f"({r},{c})  Q = [{qs[0]:6.2f} {qs[1]:6.2f} {qs[2]:6.2f} {qs[3]:6.2f}]  лучшее: {best}")

print(f"\nДлина найденного пути: {len(path) - 1} шагов")