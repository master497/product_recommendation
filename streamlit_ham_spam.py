import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# 1. Read data
data = pd.read_csv("Danh_gia_final.csv", encoding='latin-1')
# 2. Data pre-processing
source = data['v2']
target = data['v1']
# ham = 0, spam = 1
target = target.replace("Neutral", 0)
target = target.replace("Positive", 1)
target = target.replace("Negative", -1)

text_data = np.array(source)

count = CountVectorizer(max_features=6000)
count.fit(text_data)
bag_of_words = count.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)

# 3. Build model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

clf = MultinomialNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#4. Evaluate model
score_train = model.score(X_train,y_train)
score_test = model.score(X_test,y_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cr = classification_report(y_test, y_pred)
data_dc_cung_cap['noi_dung_binh_luan'] = data_dc_cung_cap['noi_dung_binh_luan'].str.replace(',', ' ').replace('.', ' ') # Xử lý dấu phẩy và dấu chấm trong trường hợp không có khoảng trống
data_dc_cung_cap['noi_dung_binh_luan'] = data_dc_cung_cap['noi_dung_binh_luan'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
data_dc_cung_cap = data_dc_cung_cap.drop_duplicates(subset=cot_can_loai_bo_trung, keep='first') # xử lý dữ liệu trùng
data_tu_cao_dieu_lieu['noi_dung_binh_luan'] = data_tu_cao_dieu_lieu['noi_dung_binh_luan'].str.replace(',', ' ').replace('.', ' ') ## Xử lý dấu phẩy và dấu chấm trong trường hợp không có khoảng trống
data_tu_cao_dieu_lieu['noi_dung_binh_luan'] = data_tu_cao_dieu_lieu['noi_dung_binh_luan'].apply(lambda x: re.sub(r'[^\w\s]', '', x)) # loại bỏ kí tự đặc biệt
emoji_dict, teen_dict, wrong_lst, stopwords_lst = load_raw_files()
data['noi_dung_binh_luan_sau_xu_ly'] = data['noi_dung_binh_luan'].apply(
    lambda x: process_text(x, emoji_dict, teen_dict, wrong_lst)
)
data['noi_dung_binh_luan_sau_xu_ly'] = data['noi_dung_binh_luan_sau_xu_ly'].apply(
    lambda x: covert_unicode(x)
)
data['noi_dung_binh_luan_sau_xu_ly'] = data['noi_dung_binh_luan_sau_xu_ly'].apply(
    lambda x: process_special_word(x)
)
data['noi_dung_binh_luan_sau_xu_ly'] = data['noi_dung_binh_luan_sau_xu_ly'].apply(
    lambda x: normalize_repeated_characters(x)
)
data['noi_dung_binh_luan_sau_xu_ly'] = data['noi_dung_binh_luan_sau_xu_ly'].apply(
    lambda x: process_postag_thesea(x)
)
data['noi_dung_binh_luan_sau_xu_ly'] = data['noi_dung_binh_luan_sau_xu_ly'].apply(
    lambda x: remove_stopword(x, stopwords_lst)
)
y_prob = model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])

model = LogisticRegression()
model.fit(X, labels)

#5. Save models
# luu model classication
pkl_filename = "sentiment_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  
# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)

#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    ham_spam_model = pickle.load(file)
# doc model count len
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)

#--------------
# GUI
st.title("Data Science Project")
st.write("## Ham vs Spam")

menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Phan Thanh Sang & Tạ Quang Hưng""")
st.sidebar.write("""#### Giảng viên hướng dẫn: """)
st.sidebar.write("""#### Thời gian thực hiện: 15/12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Classifying customer feedback is one of the most common natural language processing tasks. With the advancements in machine learning and natural language processing techniques, it is now possible to separate spam messages from ham messages with a high degree of accuracy.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for mes classification.""")
    st.image("ham_spam.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data[['v2', 'v1']].head(3))
    st.dataframe(data[['v2', 'v1']].tail(3))  
    st.write("##### 2. Visualize Ham and Spam")
    fig1 = sns.countplot(data=data[['v1']], x='v1')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()       
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for Ham vs Spam classification.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            x_new = count_model.transform(lines)        
            y_pred_new = ham_spam_model.predict(x_new)       
            st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new)) 

