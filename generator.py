import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def generate():
    A0 = 45
    A1 = 20
    A2 = 10
    A3 = 5
    A4 = 20
    Lambda = 50
    B = 1.3 * (A0 + A1 + A2 + A3 + A4)
    C = 75
    t = np.linspace(0, config.TIME_RANGE, config.TIME_NUM)  # 时间范围
    f = np.zeros_like(t)

    # 叠加波动曲线
    f += A0 * np.sin(0.25 * np.pi * t)
    f += A1 * np.sin(1 * np.pi * t - np.pi / 3)
    f += A2 * np.sin(2 * np.pi * t + np.pi / 4)
    f += A3 * np.sin(4 * np.pi * t + np.pi / 2)
    f += A4 * np.sin(1 * np.pi * t + np.pi / 6)

    # 移动到正值
    f += C
    f = f.clip(min=0, max=B)

    # 叠加随机
    f -= Lambda
    f += np.random.poisson(Lambda, f.size)

    # 限定范围至0~50
    _range = np.max(f) - np.min(f)
    f = 50 * (f - np.min(f)) / _range

    # 叠加激增
    flag = False
    cnt = 0
    for i in range(f.size):
        if random.random() < 0.001:
            flag = True
            cnt = 0
        if cnt < 10 and flag:
            f[i] += np.random.poisson(Lambda)
            cnt += 1

    # 限制范围至0~100
    _range = np.max(f) - np.min(f)
    f = 100 * (f - np.min(f)) / _range

    return t, f


if __name__ == '__main__':
    t, f = generate()
    # 绘制图像
    plt.plot(t[:500], f[:500])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform with Oscillation')
    plt.show()

    cols = ['query']
    data = pd.DataFrame(f, columns=cols)
    data.to_csv('data.csv')
