"""
奖励函数可视化
绘制奖励函数随距离变化的曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def compute_reward_components(d_T, d_O, c1=500.0, c2=15.0):
    """计算奖励组成部分"""
    R_T = 0.5 * d_T ** 2
    R_O = (1.0 / (1.0 + d_O)) ** 35
    reward = -c1 * R_T - c2 * R_O
    return reward, R_T, R_O


def plot_obstacle_penalty():
    """绘制障碍物惩罚曲线"""
    # 距离范围: 0.001m 到 0.5m
    d_O = np.linspace(0.001, 0.5, 500)
    R_O = (1.0 / (1.0 + d_O)) ** 35

    plt.figure(figsize=(12, 5))

    # 子图1: 线性尺度
    plt.subplot(1, 2, 1)
    plt.plot(d_O * 100, R_O, 'r-', linewidth=2)
    plt.xlabel('障碍物距离 d_O (cm)', fontsize=12)
    plt.ylabel('R_O = (1/(1+d_O))^35', fontsize=12)
    plt.title('障碍物惩罚函数 (线性尺度)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='R_O=0.1')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='R_O=0.5')
    plt.legend()

    # 标注关键点
    key_distances = [0.001, 0.01, 0.05, 0.1, 0.2]
    for d in key_distances:
        r = (1.0 / (1.0 + d)) ** 35
        plt.plot(d * 100, r, 'ro', markersize=8)
        plt.annotate(f'{d*100:.1f}cm\nR_O={r:.3f}',
                    xy=(d * 100, r),
                    xytext=(d * 100 + 2, r + 0.05),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # 子图2: 对数尺度
    plt.subplot(1, 2, 2)
    plt.semilogy(d_O * 100, R_O, 'r-', linewidth=2)
    plt.xlabel('障碍物距离 d_O (cm)', fontsize=12)
    plt.ylabel('R_O (对数尺度)', fontsize=12)
    plt.title('障碍物惩罚函数 (对数尺度)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('reward_obstacle_penalty.png', dpi=150)
    print("✓ 已保存: reward_obstacle_penalty.png")
    plt.show()


def plot_target_reward():
    """绘制目标距离奖励曲线"""
    # 距离范围: 0 到 1m
    d_T = np.linspace(0, 1.0, 500)
    R_T = 0.5 * d_T ** 2

    plt.figure(figsize=(10, 6))
    plt.plot(d_T * 100, R_T, 'b-', linewidth=2)
    plt.xlabel('目标距离 d_T (cm)', fontsize=12)
    plt.ylabel('R_T = 1/2 * d_T^2', fontsize=12)
    plt.title('目标距离奖励函数', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 标注关键点
    key_distances = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for d in key_distances:
        r = 0.5 * d ** 2
        plt.plot(d * 100, r, 'bo', markersize=8)
        plt.annotate(f'{d*100:.0f}cm\nR_T={r:.4f}',
                    xy=(d * 100, r),
                    xytext=(d * 100 + 5, r + 0.02),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('reward_target_distance.png', dpi=150)
    print("✓ 已保存: reward_target_distance.png")
    plt.show()


def plot_combined_reward():
    """绘制组合奖励曲线 (固定一个变量)"""
    c1 = 500.0
    c2 = 15.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 固定 d_O = 0.2m, 改变 d_T
    ax = axes[0, 0]
    d_T = np.linspace(0, 1.0, 500)
    d_O = 0.2
    rewards = []
    for dt in d_T:
        r, _, _ = compute_reward_components(dt, d_O, c1, c2)
        rewards.append(r)

    ax.plot(d_T * 100, rewards, 'g-', linewidth=2)
    ax.set_xlabel('目标距离 d_T (cm)', fontsize=11)
    ax.set_ylabel('总奖励 R', fontsize=11)
    ax.set_title(f'固定 d_O = {d_O*100:.0f}cm', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 子图2: 固定 d_O = 0.05m, 改变 d_T
    ax = axes[0, 1]
    d_O = 0.05
    rewards = []
    for dt in d_T:
        r, _, _ = compute_reward_components(dt, d_O, c1, c2)
        rewards.append(r)

    ax.plot(d_T * 100, rewards, 'orange', linewidth=2)
    ax.set_xlabel('目标距离 d_T (cm)', fontsize=11)
    ax.set_ylabel('总奖励 R', fontsize=11)
    ax.set_title(f'固定 d_O = {d_O*100:.0f}cm (较近)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 子图3: 固定 d_T = 0.1m, 改变 d_O
    ax = axes[1, 0]
    d_O = np.linspace(0.001, 0.5, 500)
    d_T = 0.1
    rewards = []
    for do in d_O:
        r, _, _ = compute_reward_components(d_T, do, c1, c2)
        rewards.append(r)

    ax.plot(d_O * 100, rewards, 'purple', linewidth=2)
    ax.set_xlabel('障碍物距离 d_O (cm)', fontsize=11)
    ax.set_ylabel('总奖励 R', fontsize=11)
    ax.set_title(f'固定 d_T = {d_T*100:.0f}cm', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 子图4: 固定 d_T = 0.5m, 改变 d_O
    ax = axes[1, 1]
    d_T = 0.5
    rewards = []
    for do in d_O:
        r, _, _ = compute_reward_components(d_T, do, c1, c2)
        rewards.append(r)

    ax.plot(d_O * 100, rewards, 'brown', linewidth=2)
    ax.set_xlabel('障碍物距离 d_O (cm)', fontsize=11)
    ax.set_ylabel('总奖励 R', fontsize=11)
    ax.set_title(f'固定 d_T = {d_T*100:.0f}cm (较远)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('reward_combined.png', dpi=150)
    print("✓ 已保存: reward_combined.png")
    plt.show()


def plot_3d_surface():
    """绘制3D奖励曲面"""
    c1 = 500.0
    c2 = 15.0

    # 创建网格
    d_T = np.linspace(0.01, 0.5, 100)
    d_O = np.linspace(0.01, 0.3, 100)
    D_T, D_O = np.meshgrid(d_T, d_O)

    # 计算奖励
    R_T = 0.5 * D_T ** 2
    R_O = (1.0 / (1.0 + D_O)) ** 35
    Reward = -c1 * R_T - c2 * R_O

    # 绘制3D曲面
    fig = plt.figure(figsize=(14, 6))

    # 子图1: 3D曲面
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(D_T * 100, D_O * 100, Reward,
                           cmap=cm.coolwarm, alpha=0.8,
                           linewidth=0, antialiased=True)
    ax1.set_xlabel('目标距离 d_T (cm)', fontsize=10)
    ax1.set_ylabel('障碍物距离 d_O (cm)', fontsize=10)
    ax1.set_zlabel('总奖励 R', fontsize=10)
    ax1.set_title('奖励函数 3D 曲面', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 子图2: 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(D_T * 100, D_O * 100, Reward, levels=20, cmap=cm.coolwarm)
    ax2.set_xlabel('目标距离 d_T (cm)', fontsize=11)
    ax2.set_ylabel('障碍物距离 d_O (cm)', fontsize=11)
    ax2.set_title('奖励函数等高线图', fontsize=12, fontweight='bold')
    fig.colorbar(contour, ax=ax2)

    # 添加一些关键点
    key_points = [
        (0.01, 0.2, '接近目标\n远离障碍物'),
        (0.4, 0.2, '远离目标\n远离障碍物'),
        (0.01, 0.02, '接近目标\n接近障碍物'),
        (0.4, 0.02, '远离目标\n接近障碍物'),
    ]

    for dt, do, label in key_points:
        r, _, _ = compute_reward_components(dt, do, c1, c2)
        ax2.plot(dt * 100, do * 100, 'wo', markersize=8, markeredgecolor='black', markeredgewidth=2)
        ax2.annotate(f'{label}\nR={r:.1f}',
                    xy=(dt * 100, do * 100),
                    xytext=(dt * 100 + 5, do * 100 + 3),
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig('reward_3d_surface.png', dpi=150)
    print("✓ 已保存: reward_3d_surface.png")
    plt.show()


def plot_reward_ratio():
    """绘制两个奖励项的比例"""
    c1 = 500.0
    c2 = 15.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 不同 d_T 下的贡献比例
    ax = axes[0]
    d_T_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    d_O = np.linspace(0.001, 0.3, 300)

    for dt in d_T_values:
        ratios = []
        for do in d_O:
            _, R_T, R_O = compute_reward_components(dt, do, c1, c2)
            target_contrib = c1 * R_T
            obstacle_contrib = c2 * R_O
            ratio = obstacle_contrib / (target_contrib + obstacle_contrib + 1e-10)
            ratios.append(ratio)

        ax.plot(d_O * 100, ratios, linewidth=2, label=f'd_T = {dt*100:.0f}cm')

    ax.set_xlabel('障碍物距离 d_O (cm)', fontsize=11)
    ax.set_ylabel('障碍物惩罚占比', fontsize=11)
    ax.set_title('障碍物惩罚在总惩罚中的占比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='50%')

    # 子图2: 不同 d_O 下的贡献比例
    ax = axes[1]
    d_O_values = [0.01, 0.05, 0.1, 0.2]
    d_T = np.linspace(0.01, 0.5, 300)

    for do in d_O_values:
        ratios = []
        for dt in d_T:
            _, R_T, R_O = compute_reward_components(dt, do, c1, c2)
            target_contrib = c1 * R_T
            obstacle_contrib = c2 * R_O
            ratio = target_contrib / (target_contrib + obstacle_contrib + 1e-10)
            ratios.append(ratio)

        ax.plot(d_T * 100, ratios, linewidth=2, label=f'd_O = {do*100:.0f}cm')

    ax.set_xlabel('目标距离 d_T (cm)', fontsize=11)
    ax.set_ylabel('目标距离惩罚占比', fontsize=11)
    ax.set_title('目标距离惩罚在总惩罚中的占比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='50%')

    plt.tight_layout()
    plt.savefig('reward_ratio.png', dpi=150)
    print("✓ 已保存: reward_ratio.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("奖励函数可视化")
    print("=" * 70)
    print("\n生成可视化图表...")
    print("\n提示: 关闭图表窗口以继续生成下一个图表\n")

    # 1. 障碍物惩罚曲线
    print("\n[1/5] 生成障碍物惩罚曲线...")
    plot_obstacle_penalty()

    # 2. 目标距离奖励曲线
    print("\n[2/5] 生成目标距离奖励曲线...")
    plot_target_reward()

    # 3. 组合奖励曲线
    print("\n[3/5] 生成组合奖励曲线...")
    plot_combined_reward()

    # 4. 3D曲面
    print("\n[4/5] 生成3D奖励曲面...")
    plot_3d_surface()

    # 5. 奖励比例
    print("\n[5/5] 生成奖励占比图...")
    plot_reward_ratio()

    print("\n" + "=" * 70)
    print("✓ 所有图表已生成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - reward_obstacle_penalty.png  : 障碍物惩罚曲线")
    print("  - reward_target_distance.png   : 目标距离奖励曲线")
    print("  - reward_combined.png          : 组合奖励曲线")
    print("  - reward_3d_surface.png        : 3D奖励曲面和等高线")
    print("  - reward_ratio.png             : 奖励占比分析")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
