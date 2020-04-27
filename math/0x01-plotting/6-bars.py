#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
width = 0.5
peoples = ['Farrah', 'Fred', 'Felicia']
colors =  ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits =  ['bananas', 'peaches', 'orange', 'peaches']

fig, axs = plt.subplots()
fig.suptitle("Number of Fruit per Person")

for row, couts in enumerate(fruit):
    axs.bar(peoples,
            couts,
            width,
            bottom=[sum(fruit[:row, col]) for col in range(len(couts))],
            label=fruits[row],
            color=colors[row])
    
axs.set_ylabel("Quantity of Fruit")
axs.set_ylim([0, 80])
axs.set_yticks(range(0, 81, 10))
axs.legend()

plt.show()
