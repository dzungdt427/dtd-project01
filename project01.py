# IMPORT CAC THU VIEN
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from underthesea import word_tokenize, pos_tag, sent_tokenize # sent_tokenize de phan tich cau, khong phai phan tich van ban
import regex
import string
from langdetect import detect

from googletrans import Translator
translator = Translator()
# # Test Translate English to Vietnamese
# translation = translator.translate("How are you today?", src='en', dest='vi')
# print("Original:", translation.origin)
# print("Translated:", translation.text)

# 1. TIEN XU LY DU LIEU TIENG VIET
# 1.1. Doc cac file
#LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()
#LOAD positive
file = open('files/v7_positive_VN.txt', 'r', encoding="utf8")
positive_words = file.read().split('\n')
file.close()
#LOAD negative
file = open('files/v7_negative_VN.txt', 'r', encoding="utf8")
negative_words = file.read().split('\n')
file.close()


## 1.2. Bo sung, cap nhat list tu
new_teen_dict = {'ot':'overtime','oc':'overcome','ot.':'overtime','x':'nhân','hk':'hong_kong','jp':'japan','eu':'europe','sin':'singapore',
                'cv':'công_việc',
                'bhyt':'bảo_hiểm_y_tế','bhxh':"bảo_hiểm_xã_hội",'bh':'bảo_hiểm',
                'env':'environment','env.':'environment','dev':'developer','dept':'department','dept.':'department','dev.':'developer',
                 'ko':'không','k':'không','h':'giờ',
                 'wfh':'work from home','hn':'hà_nội','hcm':'hồ_chí_minh','hcmc':'hồ_chí_minh','vn':'việt_nam',
                 'pm':'project manager','nv':'nhân_viên','tg':'thời_gian','cty':'công_ty',
                 'đc':'được','dc':'được','x2':'gấp_đôi','x3':'gấp_ba','hr':'human resources',
                 'pt':'hlv_thể_hình','ks':'khách_sạn','vp':'văn_phòng','mn':'mọi_người','sv':'sinh_viên',
                 'pc':'máy_tính_bàn','dt':'điện_thoại','pv':'phỏng_vấn','vs':'và','mội':'mọi','ngh':'nghiệm','kn':'kinh_nghiệm'}
teen_dict.update(new_teen_dict)

new_eng_to_vn_dict = {'overtime':'làm_thêm_giờ','work from home':'làm_việc_tại_nhà','hybrid':'làm_việc_linh_hoạt','match':'phù hợp','range':'dải','even':'sự_kiện',
                      'up':'tăng','ok':'tốt','no':'không',
                      'internship':'thực_tập','intern':'thực_tập_sinh','deal':'thỏa_thuận',
                      'conpensation':'lương','supportive':'hỗ_trợ','suportive':'hỗ_trợ','skills':'kỹ_năng','profile':'hồ_sơ','skill':'kỹ_năng',
                      'benefit':'lợi_ích','salary':'lương',
                      'overcome':'vượt_qua_khó_khăn','project manager':'sếp_quản_lý_dự_án','hr':'nhân_sự',
                      'environment':'môi_trường','developer':'lập_trinh_viên','department':'phòng_ban','projects':'dự_án','career':'nghề_nghiệp',
                      'opportunities':'cơ_hội','opportunity':'cơ_hội','pantry':'bữa_ăn_nhẹ','build':'xây_dựng','except':'ngoại_trừ',
                      'situation':'tình_huống','critical':'quan_trọng','care':'quan_tâm','policy':'chính_sách',
                      'interviewers':'người_phỏng_vấn','interviewer':'người_phỏng_vấn','project':'dự_án','training':'đào_tạo','management':'quản_lý'}
english_dict.update(new_eng_to_vn_dict)

for x in ['công_ty','doanh_nghiệp','cỏ','khô','tùy','hầu_như','làm_việc','công_việc','đội','đi','làm_thêm_giờ','và','lot','có','ty','công']:
  stopwords_lst.append(x)

stopwords_lst.remove('ít_khi')

positive_words.remove('')

negative_words.remove('')

# 1.3 Cac ham xu ly du lieu tho
# 1.3.1 ham process_text dung de tien xu ly van ban
def process_text(text, emoji_dict, teen_dict):
    document = text.lower() # chuyen sang chu thuong
    document = document.replace("’",'') # bo nhay don
    document = document.replace(",",' ') # bo dau phay
    document = document.replace("\n",' ') # bo xuong dong
    document = regex.sub(r'\.+', ".", document).strip() # thay the cac dau cham thanh 1 dau cham
    new_sentence =''
    for sentence in sent_tokenize(document):
        # CONVERT EMOJICON sang word
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        # CONVERT TEENCODE sang full word
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        # DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # DEL wrong words (xoa tu sai ve cu phap)
        # sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    # DEL excess blank space (xoa, chi de lai 1 khoang trang)
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

#1.3.2 Ham phan tach tieng anh, tieng viet
def check_lang(text):
    lang = detect(text)
    return lang

#1.3.3 ham dich tieng anh ra  viet
def translate_text(text, english_dict):
    new_sentence =''
    for sentence in sent_tokenize(text):
        sentence = ' '.join(english_dict[word] if word in english_dict else word for word in sentence.split())
        new_sentence = new_sentence+ sentence
    text = new_sentence
    return text

#1.3.4 Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

#1.3.5 Ham xu ly cac tu phu dinh, dac biet
def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if ('không' in text_lst) or ('khong' in text_lst) or ('chả' in text_lst) or ('chẳng' in text_lst) or ('ít' in text_lst) or ('it' in text_lst) or ('hiếm' in text_lst) or ('rất' in text_lst) or ('quá' in text_lst) or ('có' in text_lst) or ('khá' in text_lst) or ('hơi' in text_lst):
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  (word == 'không') or (word == 'khong') or (word == 'chả') or (word == 'chẳng') or (word == 'ít') or (word == 'it') or (word == 'hiếm') or (word == 'rất') or (word == 'quá') or (word == 'có') or (word == 'khá') or (word == 'hơi'):
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

#1.3.6 ham tao tu ghep tieng viet theo loai tu
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.',' ')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

#1.3.7 ham loai bo tu tieng viet thuoc stop list
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# 1.4. # hàm nhap lieu comment moi
def text_valid():
  text = input('Nhap noi dung: ')
  while True:
    if text =='' or text.isnumeric() or text.isspace():
      print('noi dung nhap khong hop le, vui long nhap lai')
      text = input('Nhap noi dung: ')
    else:
      break
  return text

# 1.5. hàm đếm từ positive, negative
def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []
    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)
    return word_count, word_list

# 1.6. Hàm truy vấn công ty
def find_id(id,data,col_id):
  detail = data[data[col_id] == int(id)]
  return detail
def find_name(name,data,col_name):
  detail = data[data[col_name] == str(name)]
  return detail


# 2. ĐỌC MODEL
import pickle
with open('sentment_analysis.pkl', 'rb') as file:  
    sentiment_model = pickle.load(file)
with open('minmaxscaler.pkl', 'rb') as file:  
    scaler = pickle.load(file)


# 3. GUI
line1 = st.sidebar.title('PROJECT01')
menu = ["Xác định vấn đề", "01 - Sentiment analysis", "02 - Information Clustering"]
choice = st.sidebar.selectbox('Menu', menu)
personal = ''' Đặng Thanh Dung \n dungdang0427@gmail.com \n DL07_K304'''
line1 = st.sidebar.write('\n')
line2 = st.sidebar.write('\n')
line3 = st.sidebar.write('\n')
line4 = st.sidebar.write('\n')
line5 = st.sidebar.write('\n')
line6 = st.sidebar.write('\n')
line6 = st.sidebar.write('\n')
line7 = st.sidebar.write('\n')
line8 = st.sidebar.write('\n')
info1 = st.sidebar.write('Thông tin học viên:')
info2 = st.sidebar.write(personal)


if choice == 'Xác định vấn đề':  
    st.title('Đồ án tốt nghiệp Data Science - Machine Learning')  
    st.subheader("Xác định vấn đề")
    st.write('Trong Project 01 có 2 vấn đề liên quan gồm:')
    st.write('#### 1. Phân tích trạng thái cảm xúc (tích cực, trung tính, tiêu cực):')
    st.write("""
        - Dữ liệu gốc: các đánh giá (reviews) về công ty của các ứng viên/ nhân viên đăng tải trên ITViec.\n
        - Tiền xử lý dữ liệu: áp dụng các thư viện, công cụ phù hợp để xử lý ngôn ngữ (tiếng Anh, tiếng Việt).\n
        - Dữ liệu đầu vào: là các đánh giá đã được tiền xử lý dữ liệu và có ý nghĩa cả về ngôn ngữ và máy học. Từ đó, tạo ra các đặc trưng (features) làm đầu vào cho các mô hình Machine learning.\n
        - Thuật toán: sử dụng Navie Bayes, Logistic regression, KNN, Decision tree,  Random forest, Ada Boost để tiến hành phân tích.\n
        - Kết quả: 
            + Lựa chọn mô hình phù hợp với dữ liệu, có thể dự đoán được các review thuộc trạng thái cảm xúc nào khi áp dụng mô hình.\n
            + Khi nhập review mới có thể xác định được trạng thái cảm xúc cụ thể.
    """)
    st.image('Sentiment_p1.png',width=300,caption='')  

    st.write('#### 2. Phân cụm/ phân nhóm các công ty:')
    st.write("""
        - Dữ liệu gốc: các đánh giá (reviews) của các ứng viên/nhân viên đã qua tiền xử lý dữ liệu từ vấn đề 1.\n
        - Tiền xử lý dữ liệu: áp dụng các thư viện, công cụ phù hợp để tạo ra các đặc trưng (features) cho mô hình.\n
        - Dữ liệu đầu vào: là các đánh giá đã được tiền xử lý dữ liệu và có ý nghĩa cả về ngôn ngữ và máy học. Từ đó, tạo ra các đặc trưng (features) làm đầu vào cho các mô hình Machine learning.\n
        - Thuật toán: sử dụng Kmeans clustering, Aglomerate clustering để tiến hành phân tích.\n
        - Kết quả: Lựa chọn mô hình phù hợp, kết quả phân cụm mang lại thể hiện được sự rõ ràng, mang đặc trưng riêng của mỗi cụm.\n
    """)
    st.image('Clustering_p1.png',width=300,caption='')   

    st.write('#### Kết quả khác:')
    st.write("""
    Bên cạnh đó, do dữ liệu đầu vào của Project01 là ngôn ngữ, tác giả tạo mới:\n
    - 04 file chứa các từ ngữ/biểu tượng thể hiện sắc thái cảm xúc tích cực/tiêu cực.\n
    - 01 file chứa kết quả tiền xử lý dữ liệu để phục vụ cho các yêu cầu người dùng sau này.\n
    """)

elif choice == '01 - Sentiment analysis':
    st.subheader("01 - Sentiment analysis")


    st.write("#### 1. Tổng quan")
    df_reviews = pd.read_csv('reviews.csv')
    
    plt.figure(figsize=(6,3))
    sns.countplot(x='Rating', data=df_reviews, hue='Rating')
    plt.title('Thống kê số lượng reviews ở mỗi điểm rating')
    st.pyplot(plt)
    st.write('Nhận xét: Đa số reviews có điểm rating >=4, số lượng reviews có điểm rating <3 chiếm khá ít.')

    describe_type = st.radio("## Chọn loại thống kê mô tả:", ("Tất cả", "Theo công ty"))
    if describe_type =='Tất cả':
        st.write(df_reviews[['Rating','Salary & benefits','Training & learning','Management cares about me','Culture & fun','Office & workspace']].describe())
    else:
        st.write(df_reviews.groupby('Company Name')[['Rating','Salary & benefits','Training & learning','Management cares about me','Culture & fun','Office & workspace']].agg(
                    avg_rating = pd.NamedAgg(column='Rating',aggfunc='mean'),
                    avg_salary_benefit = pd.NamedAgg(column='Salary & benefits',aggfunc='mean'),
                    avg_traing_learning = pd.NamedAgg(column='Training & learning',aggfunc='mean'),
                    avg_management_care = pd.NamedAgg(column='Management cares about me',aggfunc='mean'),
                    avg_culture_fun = pd.NamedAgg(column='Culture & fun',aggfunc='mean'),
                    avg_office_workspace = pd.NamedAgg(column='Office & workspace',aggfunc='mean')).sort_values(by='Company Name',ascending=True))
  
    plt.figure(figsize=(12,6))    
    sns.countplot(data = df_reviews,x='Recommend',hue='Rating')
    plt.title('Thống kê số lượng reviews ở mỗi điểm rating của trường hợp đề xuất (1) hoặc không đề xuất (0)')
    st.pyplot(plt)  
    st.write("""Nhận xét: Có thể thấy trong dữ liệu đầu vào, một số reviews có sự mâu thuẫn giữa việc đề xuất giới thiệu công ty và điểm rating ứng viên đánh giá.""")


    st.write("#### 2. Biến input và biến output")
    st.write("##### 2.1. Biến input")
    st.write("Tác giả tạo mới 02 features là 'positive_count' và 'negative_count' bằng cách đếm số lượng các từ tích cực/tiêu cực của mỗi reviews dựa trên danh sách các file lưu từ tích cực/tiêu cực tại quá trình tiền xử lý.\n")
    fig,axes = plt.subplots(2,2,figsize=(12,6))
    sns.kdeplot(data = df_reviews,x='positive_count',ax=axes[0,0])
    sns.boxplot(data = df_reviews,x='Rating',y='positive_count',ax=axes[0,1])
    sns.kdeplot(data = df_reviews,x='negative_count',ax=axes[1,0])
    sns.boxplot(data = df_reviews,x='Rating',y='negative_count',ax=axes[1,1])
    plt.tight_layout()
    st.pyplot(plt)
    st.write("Nhận xét: 2 features trên có dữ liệu phân phối lệch phải và có nhiều outlier ==> Tác giả sử dụng Minmax scaler để chuẩn hóa dữ liệu")
    st.write("""Ngoài ra, khi so sánh hiệu A = (positive_count - negative_count) và rating của từng reviews, tác giả nhận thấy, có một số review bị mâu thuẫn khi hiệu A >0 nhưng rating <3 hoặc ngược lại hiệu A <0 nhưng rating >3. Do đó, tác giả chia bộ dữ liệu đầu vào làm 2 phần:\n
- Phần 1: Sử dụng cho việc xây dựng mô hình. Đây là phần có dữ liệu có rating >=3 và positive_count >= negative_count hoặc rating <3 và positive_count < negative_count\n
- Phần 2: Sử dụng cho việc dự đoán mô hình. Đây là phần dữ liệu có mâu thuẫn theo nhận xét trên.""")

    st.write("##### 2.2. Biến output")
    st.write('''Biến output = sentiment, được quy đổi từ Rating như sau:\n- Rating <0 ==> sentiment = 0\n- Rating =3 ==> sentiment = 1\n- Rating >3 ==> sentiment = 2''')
    st.image('anh_slide_v7/v7_output.png')
    st.write("Tương tự như biến input, biến output cũng được chia làm 2 phần tương ứng.")
    

    st.write("#### 3. Xây dựng mô hình và đánh giá")
    st.write("""Tác giả sử dụng 06 mô hình gồm: Navie Bayes, KNN, Logistic regression, Decision tree, Random forest, Ada boost.\n Để đánh giá hiệu quả của 06 mô hình trên, tác giả đã thực hiện đo lường thời gian và sử dụng cross validation tính điểm accuracy trung bình.\n Kết quả như các ảnh sau:""")
    st.image("anh_slide_v7/v7_sent_01.png",caption='')
    st.image("anh_slide_v7/v7_sent_02.png")
    st.write('Chi tiết cụ thể như sau:')
    st.write('-------------------------------------------------------------')
    st.write('##### Naive Bayes')
    st.image('anh_slide_v7/v7_sent_nb.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.7929191716766867\n
            - f1_score: 0.7013375981968906\n
            - precision_score: 0.628720812812443\n
            - recall_score: 0.792919171676686
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Logistic regression')
    st.image('anh_slide_v7/v7_sent_lg.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.5871743486973948\n
            - f1_score: 0.6302542506647333\n
            - precision_score: 0.766910586708291\n
            - recall_score: 0.5871743486973948
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### KNN')
    st.image('anh_slide_v7/v7_sent_knn.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.7668670674682698\n
            - f1_score: 0.732768447023344\n
            - precision_score: 0.7121972065364229\n
            - recall_score: 0.7668670674682698\n
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Decision tree')
    st.image('anh_slide_v7/v7_sent_dt.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.5283901135604543\n
            - f1_score: 0.5709068622468283\n
            - precision_score: 0.781472400988362\n
            - recall_score: 0.5283901135604543\n
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Random forest')
    st.image('anh_slide_v7/v7_sent_rf.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.5404141616566466\n
            - f1_score: 0.5837590046571418\n
            - precision_score: 0.7791305867897335\n
            - recall_score: 0.5404141616566466
            """)
    st.write('-------------------------------------------------------------')    
    st.write('##### Ada Boost')
    st.image('anh_slide_v7/v7_sent_ab.png')
    st.write("""Các chỉ số đánh giá mô hình:\n
            - accuracy_score: 0.7995991983967936\n
            - f1_score: 0.7137096296671993\n
            - precision_score: 0.650985000712027\n
            - recall_score: 0.7995991983967936
            """)
    st.write('-------------------------------------------------------------')  
    st.write("""Nhận xét: Từ kết quả trên, tác giả lựa chọn mô hình Ada Boost làm mô hình để dự đoán vấn đề sentiment analysis vì có điểm accuracy cao và thời gian thực hiện ngắn. Kết quả dự đoán dữ liệu phần 1 như sau:""")
    st.image('anh_slide_v7/v7_sent_kq1.PNG')
    st.write('Dữ liệu phần 2 được sử dụng để dự đoán trạng thái cảm xúc. Kết quả dự đoán như sau:')
    st.image('anh_slide_v7/v7_sent_kq2.PNG')


    st.write("#### 4. Dự đoán")
    type = st.checkbox("Nhập dữ liệu")
    if type:        
        comment = st.text_area(label="Nhập nội dung:")
    submit = st.button("Submit")    
    if submit:  
        comment = process_text(str(comment), emoji_dict, teen_dict)
        if check_lang(str(comment)) == 'vi':
            comment = translate_text(str(comment),english_dict)
        else:
            translation = translator.translate(comment, src=check_lang(str(comment)), dest='vi')
            comment = translation.text
        comment = covert_unicode(str(comment))
        comment = process_postag_thesea(str(comment))
        comment = remove_stopword(str(comment), stopwords_lst) 
        # st.write("Nôi dung đã tiền xử lý tiếng việt:")
        # st.write(comment)
        positive = find_words(str(comment), positive_words)[0]
        negative = find_words(str(comment), negative_words)[0]
        # st.write('positive: '+str(positive))
        # st.write('negative: '+str(negative))
        X_comment = pd.DataFrame([[positive,negative]],columns=['positive_count','negative_count'])
        # st.write(X_comment)
        X_comment = scaler.transform(X_comment)        
        y_pred_new = int(sentiment_model.predict(X_comment)[0])
        if y_pred_new ==2:
            st.write('Kết quả dự đoán: '+str(y_pred_new) + ' - Tích cực')
        elif y_pred_new ==1:
            st.write('Kết quả dự đoán: '+str(y_pred_new) + ' - Trung tính')   
        else:
             st.write('Kết quả dự đoán: '+str(y_pred_new) + ' - Tiêu cực')  
        # xac_suat = max(sentiment_model.predict_proba(X_comment)[0])
        # st.write('Xác suất: '+str(xac_suat))     

elif choice == '02 - Information Clustering':
    st.subheader("02 - Information Clustering")

    st.write("#### 1. Tổng quan")
    df_cluster = pd.read_csv('data.csv')
    df_cluster = df_cluster.drop(['noi_dung_new','prediction', 'company_info'],axis=1)
    st.write('Căn cứ thông tin mô tả các công ty trên trang ITViec với bức tranh tổng quan như sau:')

    plt.figure(figsize=(12,3))
    plt.subplot(1,2,1)
    plt.title('Loại công ty')
    sns.countplot(data = df_cluster,x='company_type',hue='company_type')
    plt.xticks(rotation=90)
    plt.subplot(1,2,2)
    plt.title('Quy mô')
    sns.countplot(data = df_cluster,x='company_size',hue='company_size')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.write('-----------------------------------------------------')
    plt.figure(figsize=(12,3))
    plt.title('Quốc gia')
    sns.countplot(data = df_cluster,x='country',hue='country')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.write('-----------------------------------------------------')
    plt.figure(figsize=(12,3))
    plt.title('Ngành nghề')
    sns.countplot(data = df_cluster,x='company_industry',hue='company_industry',legend=False)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.write('-----------------------------------------------------')
    st.write('Nhận xét: Các công ty chủ yếu thuộc lĩnh vực IT, có quy mô vừa và nhỏ, có trụ sở tại Việt Nam')
    st.image('anh_slide_v7/v7_clus_view.png')
    st.write('Nhận xét: Bài toán phân cụm sẽ được áp dụng cho 180 công ty có reviews.')

    st.write("#### 2. Biến input")
    st.write('''Biến input được tổng hợp từ kết quả tiền xử lý dữ liệu. Tác giả thực hiện các bước biến đổi như sau:\n
- Thực hiện gộp tất cả các reviews theo công ty tạo thành biến 'noi_dung_new'
- Gộp với dữ liệu của file overview_reviews.xlsx để tạo thành bộ dữ liệu mới''')
    st.write('Tác giả sử dụng 02 mô hình phân cụm được trình bày tại mục 3.')


    st.write("#### 3. Xây dựng mô hình và đánh giá")
    st.write('##### Chọn k cụm')
    st.image('anh_slide_v7/v7_clus_k.png')
    st.write('Nhận xét: Chọn k = 3 làm số cụm vì theo hình vẽ, độ dốc tại k=3 có xu hướng giảm dần')

    st.write('##### Kmeans')
    st.image('anh_slide_v7/v7_clus_lda.png')
    st.write('Các chủ đề gồm:')
    st.image('anh_slide_v7/v7_clus_topic.PNG')
    st.write('Nhận xét: Các chủ đều phản ánh các đánh giá tích cực của ứng viên/nhân viên về lương, nơi làm việc, sếp, đồng nghiệp, phát triển. Tuy nhiên, thứ tự của cá yếu tố trên ở mỗi cụm lại có sự khác nhau, cho thấy mức độ ưu tiên trong đánh giá của mỗi ứng viên/nhân viên.')
    st.write('-----------------------------------------------------')
    st.image('anh_slide_v7/v7_clus_cum0.png')
    st.write('-----------------------------------------------------')
    st.image('anh_slide_v7/v7_clus_cum1.png')
    st.write('-----------------------------------------------------')
    st.image('anh_slide_v7/v7_clus_cum2.png')
    st.write('-----------------------------------------------------')
    st.write('Thống kê các yếu tố của từng cụm')
    st.image('anh_slide_v7/v7_clus_kq.PNG')
    st.write('-----------------------------------------------------')
    st.write('Nhận xét: Chọn Kmeans làm mô hình phân cụm vì các cụm được phân chia có thống số đặc trưng riêng và trên biểu đồ hiển thị các cụm không bị chồng chéo vào nhau.')

    st.write()
    st.write('##### Aglomerate')
    st.image('anh_slide_v7/v7_aglo_lda.png')
    st.write('Các chủ đề gồm:')
    st.image('anh_slide_v7/v7_aglo_topic.PNG')   
    st.write('Nhận xét: Các chủ đều phản ánh các đánh giá tích cực của ứng viên/nhân viên về lương, nơi làm việc, sếp, đồng nghiệp, phát triển. Tuy nhiên, thứ tự của cá yếu tố trên ở mỗi cụm lại có sự khác nhau, cho thấy mức độ ưu tiên trong đánh giá của mỗi ứng viên/nhân viên.')
    st.write('-----------------------------------------------------')
    st.image('anh_slide_v7/v7_aglo_cum0.png')
    st.write('-----------------------------------------------------')
    st.image('anh_slide_v7/v7_aglo_cum1.png')
    st.write('-----------------------------------------------------')
    st.image('anh_slide_v7/v7_aglo_cum2.png')
    st.write('-----------------------------------------------------')
    st.write('Thống kê các yếu tố của từng cụm')
    st.image('anh_slide_v7/v7_aglo_kq.PNG')
    st.write('-----------------------------------------------------')
    st.write('Nhận xét: Không chọn Aglomerate làm mô hình phân cụm vì các cụm được phân chia có thống số chưa rõ ràng và trên biểu đồ hiển thị các cụm bị chồng chéo vào nhau.')

    st.write()
    st.write('#### 4 Phân tích công ty')
    id_list = df_cluster.loc[~df_cluster['cluster'].isnull(),'id'].to_list()
    name_list = df_cluster.loc[~df_cluster['cluster'].isnull(),'company_name'].to_list()

    company_select = st.radio("Chọn tìm kiếm công ty theo", ("Id", "Tên"))
    if company_select =='Id':
        id = st.selectbox("", df_cluster['id'].to_list())
        if id in id_list:
            company_detail = find_id(id,df_cluster,'id').T
            st.write(company_detail)
        else:
            st.write('Không tìm thấy dữ liệu thỏa mãn')
    else:
        name = st.selectbox("", df_cluster['company_name'].to_list())
        if name in name_list:
            company_detail = find_name(name,df_cluster,'company_name').T
            st.write(company_detail)
        else:
            st.write('Không tìm thấy dữ liệu thỏa mãn')    