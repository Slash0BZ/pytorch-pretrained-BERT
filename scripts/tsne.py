from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from matplotlib.backends.backend_pdf import PdfPages


pca = decomposition.PCA(n_components=50)

tsne = TSNE(n_components=2, random_state=0, n_iter=5000)
lines = [x.strip() for x in open("timebank_logits.txt").readlines()]
X = []
Y = []

count_map = {}
threashold = 200
for line in lines:
    label = float(line.split("\t")[-1])
    if label not in [1.0, 8.0]:
        continue
    if label not in count_map:
        count_map[label] = 0
    count_map[label] += 1
    if count_map[label] > threashold:
        continue
    # if label == 0.0:
    #     count_0 += 1
    #     if count_0 > 200:
    #         continue
    # if label == 1.0:
    #     count_1 += 1
    #     if count_1 > 200:
    #         continue
    string_label = ""
    if label == 0.0:
        string_label = "seconds"
    if label == 1.0:
        string_label = "minutes"
    if label == 4.0:
        string_label = "days"
    if label == 5.0:
        string_label = "weeks"
    if label == 6.0:
        string_label = "months"
    if label == 7.0:
        string_label = "decades"
    if label == 8.0:
        string_label = "centuries"
    # Y.append(float(line.split("\t")[-1]))
    Y.append(string_label)
    X.append([float(x) for x in line.split("\t")[:-1]])


X_PCA = pca.fit(X)
X = pca.transform(X)
X_embedded = tsne.fit_transform(X)

palette = sns.color_palette("bright", 2)

sns.set(rc={'figure.figsize':(11.7, 8.27)})

sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=Y, palette=palette, style=Y, hue_order=["minutes", "centuries"], style_order=["centuries", "minutes"])

plt.legend(fontsize='x-large', title_fontsize='40')
plt.savefig('bert-3label-200.png', dpi=300)
