# **Project: Text Classification using Naive Bayes**  
**Text Classification** là bài toán có phân loại văn bản vào các phân lớp đã quy định sẵn.
Trong project này, chúng ta sẽ xây dựng một chương trình **Text Classification** liên quan đến việc phân loại một đoạn tin nhắn là tin nhắn spam hay không, sử dụng thuật toán **Naive Bayes**.
Theo đó, Input/Output của chương trình bao gồm:  
• **Input**: Một đoạn tin nhắn (text).  
• **Output**: Có là tin nhắn spam hay không (bool).  
Theo đó, với bộ dữ liệu tải về có nhãn về tin nhắn ***spam*** hoặc không spam (***ham***), chúng ta sẽ đưa qua một số bước tiền xử lý dữ liệu để tách ra các đặc trưng và nhãn tương ứng. Khi đã chuẩn bị bộ dữ liệu cho việc huấn luyện, ta thực hiện xây dựng mô hình Naive Bayes Classifier. Cuối cùng, sử dụng mô hình Naive Bayes đã huấn luyện được, ta có thể dự đoán một tin nhắn bất kì có là spam hay không.  
**Bộ dữ liệu** sẽ gồm có 2 cột:
1. Category: gồm 2 nhãn là Ham và Spam, với ý nghĩa như sau:  
• Ham: Là những tin nhắn bình thường, không có mục đích quảng cáo hoặc lừa
đảo hoặc nói cách khác là người nhận mong muốn nhận được.  
• Spam: Là những tin nhắn không mong muốn, thường có mục đích quảng cáo sản
phẩm, dịch vụ, hoặc lừa đảo.
2. Message: là những nội dung bên trong một Messages.  

Nhiệm vụ của chúng ta là dựa vào nội dung Message để phân loại nhị phân với Naive Bayes, để xem xét rằng, liệu với nội dung như thế này thì Message đó là ***Spam*** hay ***Ham***.

## **0. Tải bộ dữ liệu**
Sử dụng lệnh gdown để tải bộ dữ liệu:
```
# https://drive.google.com/file/d/1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R/view?usp=sharing
!gdown --id 1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R
```

Nếu bị giới hạn số lượt tải, hãy tải bộ dữ liệu thủ công và upload lên google drive của mình. Sau đó, sử dụng lệnh dưới đây để copy file dữ liệu vào colab:
```
from google.colab import drive

drive.mount('/content/drive')
!cp /path/to/dataset/on/your/drive .
```

## **1. Import các thư viện cần thiết**
```
import string   # Cung cấp các hàm cơ bản để thao tác với chuỗi ký tự.
import nltk     # thư viện xử lý ngôn ngữ tự nhiên phổ biến nhất trong Python.
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd # Cung cấp các cấu trúc dữ liệu hiệu quả và các công cụ để làm việc với dữ liệu.
import numpy as np  # Cung cấp các đối tượng mảng đa chiều và các hàm toán học để làm việc với các mảng này
import matplotlib.pyplot as plt

# scikit-learn: Thư viện học máy phổ biến, giúp xây dựng và triển khai các mô hình học máy phức tạp một cách nhanh chóng.
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
```

## **2. Đọc bộ dữ liệu**
```
DATASET_PATH = '/content/2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)
df
```
Để tách riêng biệt phần đặc trưng và nhãn: đọc và lưu trữ dữ liệu của từng cột vào 2 biến tương ứng messages và labels:
```
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()
print(messages[:5])
print(labels[:5])
```

##**3. Chuẩn bị bộ dữ liệu**
**3.1. Xử lý dữ liệu đặc trưng:**  

1. Chuyển đổi tất cả văn bản thành chữ thường
2. Loại bỏ tất cả các dấu câu
3. Chia văn bản thành các từ riêng lẻ
4. Loại bỏ những từ không mang ý nghĩa quan trọng
5. Rút gọn các từ thành dạng gốc của chúng, nhóm các từ tương tự lại với nhau.

*(Perform the following Preprocessing steps for the feature data: Convert all text to lowercase. Eliminates all punctuation mark. Splits the text into individual words (tokens). Filters out common words that do not carry significant meaning. Reduces words to their root form, grouping similar words together)*  


```
def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens

messages = [preprocess_text(message) for message in messages]
```

6. Tạo một bộ **Từ Điển** (Dictionary), chứa tất các từ hoặc ký tự
có xuất hiện trong toàn bộ Messages sau khi được tiền xử lý và không tính trùng lặp.

```
def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)
    return dictionary
```
7. Kế đến, chúng ta cần tạo ra những **đặc trưng** đại diện cho thông tin (là các từ) của các Message: là dựa vào ***tần suất xuất hiện*** của từ. Với mỗi Message, vector đại diện sẽ có kích thước bằng với số lượng từ có trong Dictionary.

```
def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features

dictionary = create_dictionary(messages)
X = np.array([create_features(tokens, dictionary) for tokens in messages])
```

**3.2. Xử lý dữ liệu nhãn:**  

Tiền xử lý dữ liệu nhãn bằng cách chuyển 2 nhãn ***ham*** và ***spam*** thành các con số 0 và 1 để máy tính có thể hiểu.

```
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')
```

**3.3. Chia dữ liệu train/val/test:**  

Khi tiến hành huấn luyện một mô hình machine learning, ta sẽ tách bộ dữ liệu ra thành 3 phần: Train, Validation và Test theo tỉ lệ lần lượt là 7/2/1 (trên tỉ lệ 100% của bộ dữ liệu gốc). Ngoài ra, chúng ta thêm tham số SEED để duy trì kết quả giống nhau sau mỗi lần chạy lại.
```
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)
```


##**4. Huấn luyện mô hình:**
Tạo ra 2 Input cần thiết, truyền chúng vào mô hình Gaussian Naive Bayes và tiến hành huấn luyện bằng các hàm trong thư viện sklearn.

```
%%time
model = GaussianNB()
print('Start training...')
model = model.fit(X_train, y_train)
print('Training completed!')
```

##**5. Đánh giá mô hình:**  
Sau khi huấn luyện, chúng ta đến phần đánh giá hiệu suất của mô hình. Bắt đầu với việc cho mô hình đã huấn luyện dự đoán trên tập Validation và Test.
Sau đó, sử dụng độ đo Accuracy Score để đánh giá mô hình.

```
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')
```

##**6. Thực hiện dự đoán:**
Để sử dụng mô hình cho các Message mới, chúng ta sẽ phải thực hiện lại các công đoạn Tiền xử lý, tạo đặc trưng cho Message mới này và truyền vào mô hình Naive Bayes. Lúc này, mô hình sẽ trả về giá trị 0 hoặc 1, do đó, cần gọi hàm **inverse_transform()** để chuyển đổi lại về nhãn ban đầu là ***Ham*** hoặc ***Spam***.

```
def predict(text, model, dictionary):
    processed_text = preprocess_text(text)
    features = create_features(text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]

    return prediction_cls
```
```
test_input = 'I am actually thinking a way of doing something useful'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}') # output: ham
```
```
test_input = "URGENT! You have won a 1 week FREE membership in our Â£100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}') # output: spam
```

