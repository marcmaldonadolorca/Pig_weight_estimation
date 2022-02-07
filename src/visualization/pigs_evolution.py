import pandas as pd
import matplotlib.pyplot as plt


#Read weights
data = pd.read_csv('data/processed/weights_predicted.csv')
xips_names = list(set(data['chip'].tolist()))
df_lists = []

#Order by date
data = data.sort_values(by=['time'])
n_samples = []

#read pig by pig
xips_names = list(set(data['chip'].tolist()))
for i in xips_names:
    df = data[data['chip'] == i]
    y = df['real'].tolist()
    y2 = df['pred'].tolist()
    y2mean = []
    y3 = df['pred_lin'].tolist()
    y3mean = []

    #Moving average
    for j in range(len(y)):
        n = 5
        if j < n:
            n = j
        if n != 0:
            y2mean.append(sum(y2[(j-n):j]) / n)
            y3mean.append(sum(y3[(j-n):j]) / n)
        else:
            y2mean.append(y2[0])
            y3mean.append(y3[0])


    #Plot Vaules
    plt.xlabel('weight(kg)')
    plt.ylabel('sample')
    x = range(len(y))
    plt.clf()
    plt.ylim(0, 170)
    plt.title(str(i))
    plt.plot(x, y, color="black")
    plt.plot(x, y2mean, color="red")
    plt.plot(x, y3mean, color="tan")
    plt.legend(['Real', 'Predicted CNN', 'Predicted Linear Regressor'])
    plt.savefig('data/processed/plots/' + str(i) + '.png')
    n_samples.append(len(y))
    if len(y) > 30:
        print(i, len(y))


#Plot loss evolution
plt.clf()
x = [0.2,0.4,0.6,0.9,0.95]
ytrain = [48.7428, 41.2608, 22.1453, 27.4846, 24.8199]
yval = [68.7805, 48.4661,26.4691, 26.0775, 13.6710]

plt.title('Loss evolution')
plt.xlabel('training ratio')
plt.ylabel('loss')
plt.plot(x, ytrain)
plt.plot(x, yval, color="red")
plt.legend(['Train loss', 'Val loss'])
plt.savefig('loss_evolution')

plt.clf()
plt.title('Pig sample frequency')
plt.bar(range(len(n_samples)), n_samples, align='center')
plt.xlabel('bin')
plt.ylabel('frequency')
plt.savefig('histogram')
