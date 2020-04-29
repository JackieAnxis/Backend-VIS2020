import csv
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")

data = []
name = 'Motorbikes' # 'Cars'
for k in [0, 4, 8, 12, 16 ,20]:
    with open('accuracy_' + name + '_' + str(k) + '.csv') as f:
        spamreader = csv.reader(f, delimiter=',')
        i = 0
        j2m = {}
        for row in spamreader:
            if i == 0:
                j = 0
                for m in row:
                    j2m[j] = m
                    j += 1
            else:
                j = 0
                for acc in row:
                    data.append([j2m[j], k, float(acc)])
                    j += 1
            i += 1

with open('acc_' + name + '.csv', 'w') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['method', 'outlier', 'acc'])
    for row in data:
        spamwriter.writerow(row)

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
acc_moto = pd.read_csv('./acc_' + name + '.csv')
# Plot the responses for different events and regions
# sns.lineplot(x="outlier", y="acc",
#              hue="method", ci='sd',
#              data=acc_moto)
sns.relplot(x="outlier", y="acc", hue="method", kind="line", data=acc_moto)
print('xxx')