# -*- encoding: utf-8 -*-

import math
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt


# Функция гауссового импульса
def gauss(q, m, d_g, w_g, d_t, eps=1, mu=1, Sc=1):
    return np.exp(-((((q - m*np.sqrt(eps*mu)/Sc)
                      - (d_g / d_t)) / (w_g / dt)) ** 2))


# Функция - дискретизатор
def sampler(obj, d_obj: float) -> int:
    return math.floor(obj / d_obj + 0.5)


# Параметры моделирования
# Волновое сопротивление свободного пространства
W0 = 120.0 * np.pi

# Число Куранта
Sc = 1.0

# Скорость света
c = 299792458.0

# Размер области моделирования в метрах
maxSize_m = 0.7
# Дискрет по пространству
dx = 1e-3
# Размер облести моделирования в отсчетах
maxSize = math.floor(maxSize_m / dx + 0.5)

# Параметры слоев
# Отступ до начала слоев
d0_m = 0.1
# Толщина первого слоя
d1_m = 0.24
# Толщина второго слоя
d2_m = 0.14
# Отсчет начала первого слоя
posL1min = sampler(d0_m, dx)
# Отсчет начала второго слоя
posL2min = sampler(d0_m + d1_m, dx)
# Отсчет начала третьего слоя
posL3min = sampler(d0_m + d1_m + d2_m, dx)

# Параметры среды
# Диэлектрические проницаемости
eps = np.ones(maxSize)
eps[posL1min:posL2min] = 11.8
eps[posL2min:posL3min] = 4.7
eps[posL3min:] = 8.5
# Магнитная проничаемость
mu = 1.0

# Время расчета в секундах
maxTime_s = 200e-9
# Дискрет по времени
dt = Sc * dx / c
# Время расчета в отсчетах
maxTime = sampler(maxTime_s, dt)
# временна сетка
tlist = np.arange(0, maxTime * dt, dt)

# Шаг по частоте
df = 1.0 / (maxTime * dt)
# Частотная сетка
flist = np.arange(-maxTime / 2 * df, maxTime / 2 * df, df)

# Параметры гауссова сигнала
# Уровень ослабления в начальный момент времени
A_0 = 10
# Уровень ослабления на частоте F_max
A_max = 70
# Ширина спектра по уровню 0.01
F_max = 3e9
# Параметр "ширины" импульса
wg = np.sqrt(np.log(A_max)) / (np.pi * F_max)
# Параметр "задержки" импульса
dg = wg * np.sqrt(np.log(A_0))

# Положение источника в метрах
sourcePos_m = 0.08
# Положение источника в отсчетах
sourcePos = math.floor(sourcePos_m / dx + 0.5)

# Положение датчика в метрах
probe1Pos_m = 0.04
# Положение датчика в отсчетах
probe1Pos = sampler(probe1Pos_m, dx)
# Инициализация датчика
probe1Ez = np.zeros(maxTime)

# Инициализация полей
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize - 1)
# Массив, содержащий падающий сигнал
Ez0 = np.zeros(maxTime)

# Вспомогательные коэффициенты
# для расчета граничных условий
Sc1 = Sc / np.sqrt(mu * eps)
k1 = -1 / (1 / Sc1 + 2 + Sc1)
k2 = 1 / Sc1 - 2 + Sc1
k3 = 2 * (Sc1 - 1 / Sc1)
k4 = 4 * (1 / Sc1 + Sc1)
# хранение полей за предыдущие
# отсчеты времени
# Слева
oldEzL1 = np.zeros(3)
oldEzL2 = np.zeros(3)
# Справа
oldEzR1 = np.zeros(3)
oldEzR2 = np.zeros(3)

# Рассчет полей
for t in range(1, maxTime):
    # Расчет компоненты поля H
    Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

    # Источник возбуждения с использованием метода
    # Total Field / Scattered Field
    Hy[sourcePos - 1] -= (Sc / W0) * \
                         gauss(t, sourcePos, dg, wg, dt,
                               eps=eps[sourcePos], mu=mu)

    # Расчет компоненты поля E
    Ez[1:-1] = Ez[1: -1] + \
               (Hy[1:] - Hy[: -1]) * Sc * W0 / eps[1: -1]

    # Источник возбуждения с использованием метода
    # Total Field / Scattered Field
    Ez0[t] = Sc * gauss(t + 1, sourcePos, dg, wg, dt,
                          eps=eps[sourcePos], mu=mu)
    Ez[sourcePos] += Ez0[t]

    # Граничные условия для поля E
    # Слева
    Ez[0] = (k1[0] * (k2[0] * (Ez[2] + oldEzL2[0]) +
                      k3[0] * (oldEzL1[0] + oldEzL1[2] - 
                               Ez[1] - oldEzL2[1]) -
                      k4[0] * oldEzL1[1]) - oldEzL2[2])
    oldEzL2[:] = oldEzL1[:]
    oldEzL1[:] = Ez[0: 3]
    # Справа
    Ez[-1] = (k1[-1] * (k2[-1] * (Ez[-3] + oldEzR2[-1]) +
                        k3[-1] * (oldEzR1[-1] + oldEzR1[-3] - 
                                  Ez[-2] - oldEzR2[-2]) -
                        k4[-1] * oldEzR1[-2]) - oldEzR2[-3])

    oldEzR2[:] = oldEzR1[:]
    oldEzR1[:] = Ez[-3:]

    # Регистрация поля в датчике
    probe1Ez[t] = Ez[probe1Pos]

# Расчет спектра зарегистрированного сигнала
Ez1Spec = fftshift(np.abs(fft(probe1Ez)))
Ez0Spec = fftshift(np.abs(fft(Ez0)))
Gamma = Ez1Spec / Ez0Spec

# Отображение графиков
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# Сигналы
ax1.set_xlim(0, 0.1 * maxTime * dt)
ax1.set_ylim(-0.6, 1.1)
ax1.set_xlabel('t, с')
ax1.set_ylabel('Ez, В/м')
ax1.plot(tlist, Ez0)
ax1.plot(tlist, probe1Ez)
ax1.legend(['Падающий сигнал',
           'Отраженный сигнал'],
          loc='upper right')
ax1.minorticks_on()
ax1.grid()

Fmax = 2e9
Fmin = 0

# Спектры сигналов
ax2.set_xlim( Fmin,  Fmax)
ax2.set_xlabel('f, Гц')
ax2.set_ylabel('|F{Ez}|, В*с/м')
ax2.plot(flist, Ez0Spec)
ax2.plot(flist, Ez1Spec)
ax2.legend(['Спектр падающего сигнала',
           'Спектр отраженного сигнала'],
          loc='upper right')
ax2.minorticks_on()
ax2.grid()

# Коэффициент отражения
ax3.set_xlim( Fmin, Fmax)
ax3.set_ylim(0,0.9)
ax3.set_xlabel('f, Гц')
ax3.set_ylabel('|Г|, б/р')
ax3.plot(flist, Gamma)
ax3.minorticks_on()
ax3.grid()

plt.subplots_adjust(hspace=0.5)
plt.show()

