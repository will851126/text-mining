
# coding: utf-8

# In[1]:


import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


jieba.set_dictionary('dict.txt.big')


# ### Documents

# In[3]:


documents = ['文字探勘指的是從非結構化的文字中，萃取出有用的重要資訊或知識。',
             '資訊檢索指的是從資訊資源集合獲得與資訊需求相關的資訊資源的活動。',
             '搜尋可以基於全文或其他基於內容的索引。',
             '自然語言認知和理解，讓電腦把輸入的語言變成有意思的符號和關係，然後根據目的再處理。',
             '東吳大學是以教學為主、研究兼重之綜合大學，擁有外雙溪、城中兩校區，設有人文社會、外國語文、理、法、商及巨量資料管理等六個學院，共26個系所、學位學程。']
segments = [' '.join(jieba.cut(d)) for d in documents]
print('documents', documents)
print('segments', segments)


# ### Another documents

# In[4]:


new_documents = ['詞嵌入是自然語言處理中語言模型與表徵學習技術的統稱。',
                 '東吳大學掌握趨勢，瞄準未來，跨院系合作，首創成立「巨量資料管理學院」，培養符合企業需求的巨量資料領域專業人才，期許本學院成為國內巨量資料人才的培訓基地，無縫接軌校園的人才培訓與企業的實務需求。']
new_segments = [' '.join(jieba.cut(d)) for d in new_documents]
print('new documents', new_documents)
print('new segments', new_segments)


# ## Bag-of-Word Matrix Apporach

# In[5]:


# create bag-of-word model and fit the documents
vectorizer = CountVectorizer(binary=True)
print('vectorizer', vectorizer)

vectorizer.fit(segments)
print('Number of vocabulary', len(vectorizer.vocabulary_))
print('Vocabulary', vectorizer.vocabulary_)
print('feature name', vectorizer.get_feature_names())


# In[6]:


# transform the documents
sparse_bag_of_words = vectorizer.transform(segments)
print('Sparse matrix', sparse_bag_of_words)

dense_bag_of_words = sparse_bag_of_words.toarray()
print('Dense matrix', dense_bag_of_words)

print('Matrix shape', dense_bag_of_words.shape)


# In[7]:


# transform the new documents
new_sparse_bag_of_words = vectorizer.transform(new_segments)
new_dense_bag_of_words = new_sparse_bag_of_words.toarray()
print('New dense matrix', new_dense_bag_of_words)


# ## tf Matrix Apporach

# In[8]:


# create bag-of-word model and fit the documents
vectorizer = CountVectorizer()
print('vectorizer', vectorizer)

vectorizer.fit(segments)
print('Number of vocabulary', len(vectorizer.vocabulary_))
print('Vocabulary', vectorizer.vocabulary_)
print('feature name', vectorizer.get_feature_names())


# In[9]:


# transform the documents
dense_bag_of_words = vectorizer.transform(segments).toarray()
print('Dense matrix', dense_bag_of_words)

print('Matrix shape', dense_bag_of_words.shape)


# In[10]:


# transform the new documents
new_sparse_bag_of_words = vectorizer.transform(new_segments)
new_dense_bag_of_words = new_sparse_bag_of_words.toarray()
print('New dense matrix', new_dense_bag_of_words)


# ## tf-idf Matrix Apporach

# In[11]:


# create bag-of-word model and fit the documents
vectorizer = TfidfVectorizer()
print('vectorizer', vectorizer)

vectorizer.fit(segments)
print('Number of vocabulary', len(vectorizer.vocabulary_))
print('Vocabulary', vectorizer.vocabulary_)
print('feature name', vectorizer.get_feature_names())


# In[12]:


# transform the documents
dense_bag_of_words = vectorizer.transform(segments).toarray()
print('Dense matrix', dense_bag_of_words)

print('Matrix shape', dense_bag_of_words.shape)


# In[13]:


# transform the new documents
new_sparse_bag_of_words = vectorizer.transform(new_segments)
new_dense_bag_of_words = new_sparse_bag_of_words.toarray()
print('New dense matrix', new_dense_bag_of_words)

