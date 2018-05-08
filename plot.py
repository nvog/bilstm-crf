import matplotlib.pyplot as plt

train = [7.88, 2.95, 1.59, 0.96, 0.62, 0.4, 0.26, 0.16]
dev = [4.49, 2.45, 1.89, 1.61, 1.54, 1.59, 1.81, 1.77]
name = 'curves'

plt.plot(train, marker='o', color='b', label='train')
plt.plot(dev, marker='o', color='r', label='dev')
plt.xlabel('Epochs')
plt.ylabel('Neg. Log Likelihood')
plt.xticks(range(len(train)))
plt.legend()
# plt.show()
plt.savefig('plots/{}.pdf'.format(name))