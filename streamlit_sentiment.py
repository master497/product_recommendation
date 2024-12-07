import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
import warnings
from wordcloud import WordCloud


warnings.simplefilter(action='ignore', category=FutureWarning)
@st.cache_data
def load_data():
    data = pd.read_csv("data1.csv")
    san_pham = pd.read_csv("San_pham.csv")
    return data, san_pham
data, san_pham = load_data()
# 2. Data pre-processing
source = data['noi_dung_binh_luan_sau_xu_ly']
target = data['sentiment']
# Positive = 2, Neutral = 1, Negative = 0
target = target.replace({"Positive": 2})
target = target.replace({"Neutral": 1})
target = target.replace({"Negative": 0})

text_data = np.array(source)

tfidf = TfidfVectorizer(max_features=6000)
tfidf.fit(text_data)
bag_of_words = tfidf.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)
# y = label_binarize(y, classes=[0, 1, 2])

# 3. Build model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

clf = LogisticRegression(class_weight='balanced')
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#4. Evaluate model
score_train = model.score(X_train,y_train)
score_test = model.score(X_test,y_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])


cr = classification_report(y_test, y_pred)

y_prob = model.predict_proba(X_test)
# roc = roc_auc_score(y_test, y_prob[:, 1])

#5. Save models
# luu model classication
pkl_filename = "Sentiment_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  
# luu model CountVectorizer (count)
pkl_tfidf = "tfidf_model.pkl"  
with open(pkl_tfidf, 'wb') as file:  
    pickle.dump(tfidf, file)

# idx = indices[product_name]


#5.5/ ve wordcloud
def generate_wordcloud(text, title):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)

#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    sentiment_model = pickle.load(file)
# doc model count len
with open(pkl_tfidf, 'rb') as file:  
    tfidf_model = pickle.load(file)

#--------------
# GUI
st.title("Data Science Project")
st.write("## Sentiment Model")

menu = ["Business Objective", "Build Project", "Item Code", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Phan Thanh Sang & Tạ Quang Hưng""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Cô Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện:
                 12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Classifying customer feedback is sentiment analysis. With the advancements in machine learning and natural language processing techniques, it is now possible to separate feedback with a high degree of accuracy.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for customer feedback classification.""")
    st.image("sentiment.png")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data[['noi_dung_binh_luan_sau_xu_ly', 'sentiment']].head(3))
    st.dataframe(data[['noi_dung_binh_luan_sau_xu_ly', 'sentiment']].tail(3))  
    st.write("##### 2. Visualize So Sao")
    fig1 = sns.countplot(data=data[['sentiment']], x='sentiment')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model")
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    # st.code(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Neutral', 'Negative'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)
    st.write("###### Classification report:")
    st.code(cr)
    # st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    # st.write("###### ROC curve")
    # # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    # fig, ax = plt.subplots()       
    # ax.plot([0, 1], [0, 1], linestyle='--')
    # ax.plot(fpr, tpr, marker='.')
    # st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for customer feedback classification.")

elif choice == 'Item Code':
    st.subheader("Review item")
    search_query = st.text_input("Tìm kiếm sản phẩm:", "") 
    filtered_products = san_pham[san_pham["ten_san_pham"].str.contains(search_query, case=False, na=False)] if search_query else san_pham
    if filtered_products.empty:
        st.warning("Không tìm thấy sản phẩm nào.")
    else:
        product_mapping = dict(zip(filtered_products["ten_san_pham"], filtered_products["ma_san_pham"]))
        product_name = st.selectbox("Chọn tên sản phẩm:", options=product_mapping.keys())
        # product_name = st.selectbox("Chọn sản phẩm bạn thích:", san_pham["ten_san_pham"].unique())
        product_id = product_mapping[product_name]
        
        st.write(f"### Sản phẩm được chọn: {product_name} (Mã sản phẩm: {product_id})")
        product_data = data[data["ma_san_pham"] == product_id]

        if product_data.empty:
            st.warning("Không có dữ liệu phản hồi cho sản phẩm này.")
        else:
            positive_reviews = product_data[product_data["sentiment"] == "Positive"]
            negative_reviews = product_data[product_data["sentiment"] == "Negative"]

    
            st.write("### Đánh giá từ khách hàng")
            st.write(f"#### Số nhận xét tích cực: {len(positive_reviews)}")
            st.write(f"#### Số nhận xét tiêu cực: {len(negative_reviews)}")

    
            st.write("### Word Cloud")
            if not positive_reviews.empty:
                st.write("#### Positive Reviews:")
                generate_wordcloud(positive_reviews["noi_dung_binh_luan_sau_xu_ly"], "Positive Reviews")

            if not negative_reviews.empty:
                st.write("#### Negative Reviews:")
                generate_wordcloud(negative_reviews["noi_dung_binh_luan_sau_xu_ly"], "Negative Reviews")

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
            x_new = tfidf_model.transform(lines)        
            y_pred_new = sentiment_model.predict(x_new)       
            st.code("New predictions (2: Positive, 1: Neutral, 0: Negative): " + str(y_pred_new)) 

