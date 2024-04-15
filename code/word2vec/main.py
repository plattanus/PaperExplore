import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import TfidfModel, Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from warnings import filterwarnings
filterwarnings('ignore')

with open("test.dat", "r") as file:
    content = file.read()
# print(content)

# cut_word_list = np.array([cont.split() for cont in content.tolist()])
# cut_word_list = np.array([cont.split() for cont in content.splitlines()])
cut_word_list = [cont.split() for cont in content.splitlines()]
dictionary = corpora.Dictionary(cut_word_list)
corpus = [dictionary.doc2bow(text) for text in cut_word_list]
# print(cut_word_list[14998])
# print(dictionary)
# print(corpus)

# word2vec训练词向量
def word2vec_model():
    model = Word2Vec(cut_word_list, vector_size=200, window=5, min_count=1, seed=1, workers=4)
    model.save('word2vec.model')
word2vec_model()
# 加载模型得出词向量
model = Word2Vec.load('word2vec.model')
model.train(cut_word_list, total_examples=model.corpus_count, epochs=10)
wv = model.wv  # 所有分词对应词向量
# print(model)

# word2vec构建文档向量
def get_word2vec_vec(content=None):
    # text_vec = np.zeros((content.shape[0], 200))
    text_vec = np.zeros((len(content), 200))
    for ind, text in enumerate(content):
        wlen = len(text)
        vec = np.zeros((1, 200))
        # print(text)
        for w in text:
            try:
                vec += wv[w]
                # print(len(wv[w]))
            except:
                pass
        text_vec[ind] = vec/wlen
    word2vec = pd.DataFrame(data=text_vec)
    word2vec.to_csv('word2vec.csv', index=False)
    return text_vec
    
word2vec = get_word2vec_vec(cut_word_list)
print(len(word2vec))
np.savetxt('word2vec.txt', word2vec)
# print(word2vec)


word_id = dictionary.token2id
tfidf_model = TfidfModel(corpus, normalize=False)
corpus_tfidf = [tfidf_model[doc] for doc in corpus]
corpus_id_tfidf = list(map(dict, corpus_tfidf))

def get_tfidf_vec(content=None):
    # text_vec = np.zeros((content.shape[0], 200))
    text_vec = np.zeros((len(content), 200))
    for ind, text in enumerate(content):
        wlen = len(text)
        vec = np.zeros((1, 200))
        for w in text:
            try:
                if word_id.get(w, False):
                    vec += (wv[w] * corpus_id_tfidf[ind][word_id[w]])
                    # print(len(wv[w]), corpus_id_tfidf[ind][word_id[w]])
                else:
                    vec += wv[w]
            except:
                pass
        text_vec[ind] = vec/wlen
    tfidf = pd.DataFrame(data=text_vec)
    tfidf.to_csv('tfidf_vec.csv', index=False)
    return text_vec
    
tfidf = get_tfidf_vec(cut_word_list)
print(len(tfidf))
np.savetxt('tfidf.txt', tfidf)
# print(tfidf)



def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

triple_similarity  = []
for i in range(14999):
    for j in range(14999, 29999):
        print(i,j)
        triple_similarity.append((i, j, cosine_similarity(tfidf[i], tfidf[j])))
np.savetxt('triple_similarity.txt', triple_similarity)

# 轮廓系数确定簇数 -> 最佳值为1，最差值为-1。接近0的值表示重叠的群集
def silhouette_score_show(data_vec=None, name=None):
    k = range(2, 10)
    score_list = []
    for i in k:
        model = KMeans(n_clusters=i).fit(data_vec)
        y_pre = model.labels_
        score = round(silhouette_score(data_vec, y_pre), 2)
        score_list.append(score)
    plt.figure(figsize=(12, 8))
    plt.plot(list(k), score_list)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('簇数', fontsize=15)
    plt.ylabel('系数', fontsize=15)
    plt.savefig(f'{name}轮廓系数.jpg')
    plt.show()


# silhouette_score_show(word2vec, 'word2vec')
# silhouette_score_show(tfidf, 'tfidf')


# kmeans = KMeans(n_clusters=2).fit(tfidf)
# y_pre = kmeans.labels_
# labels = pd.DataFrame(y_pre)
# print(labels.value_counts())

# data = pd.DataFrame(tfidf)
# data_label = pd.concat([data, labels], axis=1) 
# data_label.columns = [f'vec{i}' for i in range(1, tfidf.shape[1]+1)]  + ['label']
# print(data_label.head())



# tsne = TSNE()
# tsne.fit_transform(data_label) # 进行数据降维
# tsne = pd.DataFrame(tsne.embedding_,index=data_label.index) # 转换数据格式 
# plt.figure(figsize=(10, 6))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# d = tsne[data_label['label'] == 0]
# plt.plot(d[0],d[1],'rD')
# d = tsne[data_label['label'] == 1]
# plt.plot(d[0],d[1],'go')
# plt.xticks([])
# plt.yticks([])
# plt.legend(['第一类', '第二类'], fontsize=12)
# plt.savefig('文本聚类效果图.jpg')
# plt.show()


# word = "一口气"  # 要查询相似词的词汇

# 使用Word2Vec模型计算词汇之间的相似度
# similar_words = model.wv.most_similar(word)  # 假设model是已经训练好的Word2Vec模型

# print(f"The most similar words to '{word}' are:")
# for similar_word, similarity in similar_words:
#     print(f"{similar_word}: {similarity}")