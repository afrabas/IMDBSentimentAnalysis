#Kütüphaneleri tanımlıyoruz.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import model_selection
import nltk
nltk.download('stopwords')

#Verisetini tanımlıyoruz.
dataset_columns = ["review", "sentiment"]
imdb = pd.read_csv('IMDB Dataset.csv', encoding='ISO-8859-1',names=dataset_columns)

df = pd.DataFrame()
df["review"] = imdb["review"]
df["sentiment"] = imdb["sentiment"]
#preprocessing kısmı
df['review']=df['review'].apply(lambda x: " ".join(x.lower() for x in x.split())) #büyük küçük dönüşümü yaptık.
df['review']=df['review'].str.replace('[^\w\s]','') #noktalama işaretlerini kaldırdık.
df['review']=df['review'].str.replace('\d','') #sayıları kaldırdık.
df['review']=df['review'].str.replace('[^a-zA-Z#]',' ')
df['review']=df['review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))  #"hmm", "the" gibi çok kullanılan ama ihtiyacımız olmayan kelimeleri attık.
delete = pd.Series(' '.join(df['review']).split()).value_counts()[-1000:]
df['review']=df['review'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))
#stopwords silinmesi
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['review']=df['review'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

tokenized_data = df['review'].apply(lambda x: x.split())

from nltk.stem import PorterStemmer #stemming yapıyoruz. yani kelimelerin sonundaki ekleri siliyoruz.(-ing,-s,-ed gibi ekler)
stemmer = PorterStemmer()

tokenized_data = tokenized_data.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range (len(tokenized_data)):
    tokenized_data[i] = ' '.join(tokenized_data[i])
    
df['review'] = tokenized_data

def MachineLearning():
    #Verimizi train ve test olarak 0.33 oranda ayırıyoruz.
    validation_size = 0.33
    seed = 42 
    train_x, test_x, train_y, test_y = train_test_split(df['review'], df['sentiment'], 
                                                        test_size=validation_size, random_state=seed)

    plt.figure() #Bu kodları görüntü alabilmek için yazmıştım.
    sns.countplot(train_y)
    plt.xlabel("Classes")
    plt.ylabel("Freq")
    plt.title("Y Train")
    
    plt.figure()
    sns.countplot(test_y)
    plt.xlabel("Classes")
    plt.ylabel("Freq")
    plt.title("Y Test")
    
    #Encode işlemi yapıyoruz. Label Encoder veriyi birebir sayısallaştırmaya yarar. (1,2,...n şeklinde)
    encoder = preprocessing.LabelEncoder() 
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    #Tf-Idf Vectorizer kullandık, yani bir kelimenin hem ait olduğu text içerisindeki sıklığına hem de bütün textler içindeki
    #sıklığına göre bir değer elde ediyoruz. 
    tf_idf_word_vectorizer = TfidfVectorizer()
    tf_idf_word_vectorizer.fit(train_x)
    x_train_tf_idf_word= tf_idf_word_vectorizer.transform(train_x)
    x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)


    #Linear Regression ile tahmin değeri buluyoruz.
    print("Logistic Regression'a göre:")
    log = linear_model.LogisticRegression()
    log_model = log.fit(x_train_tf_idf_word, train_y)
    accuracy = model_selection.cross_val_score(log_model,
                                               x_test_tf_idf_word,
                                               test_y,
                                               cv=10).mean()
    print("Linear Regression'a göre TF-IDF Doğruluk Oranı:" ,accuracy)

    
    
    #KNN algoritmasıyla tahmin değeri buluyoruz.
    print("KNN (K-Nearest Neighborhood) Algoritmasına Göre TF-IDF Doğruluk Oranı: ")
    knn = KNeighborsClassifier() 
    knn.fit(x_train_tf_idf_word, train_y) 
    knn_prediction = knn.predict(x_test_tf_idf_word) 
    knn_cm = confusion_matrix(test_y, knn_prediction) 
    print("Confusion Matrisi:") #Hata matrisi
    print(knn_cm)
    print("KNN Classification Raporlaması: ") #Sınıflandırma raporlaması.
    print(classification_report(test_y, knn_prediction))
    plt.matshow(knn_cm)
    plt.title('KNN Hata Matrisi')
    plt.colorbar()
    plt.ylabel('Actual Label')
    plt.xlabel('Prediction Label')
    plt.show()
    
        
        
def LSTMwithKeras(): #Derin öğrenme kullanarak doğruluk değeri buluyoruz.
    #Keras kütüphanelerini tanımlıyoruz.
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Embedding, Dropout
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    
    validation_size = 0.2
    train_x, test_x, train_y, test_y = train_test_split(df['review'], df['sentiment'], test_size=validation_size, random_state=42)
    

    
    tokenizer = Tokenizer(num_words=15000, split=" ")
    tokenizer.fit_on_texts(train_x.values)
    X = tokenizer.texts_to_sequences(train_x.values)
    X_test = tokenizer.texts_to_sequences(test_x.values)
    maxlen=130
    X = pad_sequences(X, maxlen=maxlen) #Bizim reviewlarımız farklı boyutlarda, padding işlemiyle onları eşitliyoruz.
    X_test = pad_sequences(X_test, maxlen=maxlen)
    encoder = preprocessing.LabelEncoder() #Aşağıda X train ve X test değerleri sayısallaşacağı için Y'ye de aynısını uyguluyoruz.
    train_y = encoder.fit_transform(train_y) 
    test_y = encoder.fit_transform(test_y)
     
    lstm_model = Sequential() 
    lstm_model.add(Embedding(15000, 32, input_length = X.shape[1])) #Embedding integerları belirli boyutlarda yoğunluk vektörlerine çeviriyor. 15.000 da vocabulary size'ı.
    #X.shape[1]=130 aslında.
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
    lstm_model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))               
    lstm_model.add(Dense(1, activation='sigmoid')) #sigmoid çünkü binary classification yapabilmek için.
    lstm_model.compile(loss = "binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])   
    #binary_crossentropy kullandık çünkü 2 tane label classımız var. daha fazla olursa categoricalcrossentropy kullanılıyor.
    lstm_model.summary()
    #yukarıda lstm modelini inşa ettik, aşağıda da eğiteceğiz.
    batch_size=128 #train örneklerinin sayısını belirler. (1 forward/1 backward pass)
    #Epoch çok büyük verisetlerine uygulanırken batch_sizelara bölünür. Iterasyon sayısı diyebiliriz.
    epochs = 5 
    history = lstm_model.fit(X, train_y, validation_split=0.2, epochs=epochs,
                             batch_size=batch_size, verbose=1)
    predictions = lstm_model.predict(X_test)
    
    print((df['review'][i], predictions[i], test_y[i]) for i in range(0,5))
    
    score = lstm_model.evaluate(X_test, test_y)
    print("Accuracy(%): ",score[1]*100) #Doğruluk oranı
    
    plt.figure() #Rapora konmak üzere grafik oluşturuyoruz.
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Test")
    plt.title("Accuracy")
    plt.ylabel("Accuracies")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    
print("Test etmek istediğiniz yöntemi seçiniz: ")
print("1) Makine Öğrenmesi Algoritmaları(Linear Regression, KNN...) ")
print("2) LSTM")
girdi = int(input("Lütfen bir sayı giriniz: "))
while girdi not in [1, 2]:
    girdi= int(input("Lütfen sayıyı doğru giriniz: "))
if(girdi==1):
    MachineLearning()
    
elif(girdi==2):
    LSTMwithKeras()