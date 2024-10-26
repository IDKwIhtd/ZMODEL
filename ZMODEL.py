import subprocess
import sys

# 필요한 패키지 설치
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'PyKomoran'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'tomotopy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'tqdm'])
import requests

import pickle
import time
import requests
import pandas as pd
from PyKomoran import*
#komoran = Komoran("EXP")  # 형태소 분석기
komoran = Komoran("EXP")

import tomotopy as tp  # LDA 모델링
import re
import urllib.parse
import json
from tqdm import tqdm  # tqdm 라이브러리 임포트
tqdm.pandas()

# Naver API 인증 정보
client_id = "AQsXYKieKKmhXR2fyWfH"
client_secret = "DjU1_8dPyy"


######################################################################custom_dict##################################################################
word_list = ['동격렬비도','서격렬비도','수강생','자원봉사자','론디포 파크','프로야구','지명타자','대기록','6타수','6안타','3홈런','10타점', '50홈런','50도루' ,
             '9월','빅데이터','브랜드이슈','브랜드소통','브랜드확산','브랜드시장','브랜드공헌','오광환','도약점','Blackwell','야놀자','시즌권','2년','생산처',
             '김여사','프레스룸','특검법','로텐더홀','재의결','클라우디안','유윌씨','운암복합문화체육센터','전진숙','생활체육','인하대병원','군사도발','가능성',
             '국가안보실장','김 위원장','주낙영','황촌','상권활력소','빅컷','선린대','응급구조과','베테랑2','1위','노상현','김길리','토리노 동계U대회',
             '토리노동계세계대학경기대회','파견선수','오승하','축제무대','행복한 아저씨','분당의 밤','월드아트팩토리','국민의힘','친윤계','채상병 특검법', '지역화폐법','재의요구',
             '윤 대통령','김건희 특검법','배승희','농식품부','버디퍼트','노승희','88컨트리클럽','요진건설','친윤','친한','국정감사대책회의','김여사','26일','14년','조선중앙TV',
             '진영외교','심현근','부장판사','25일','4학년','우크라전','본격화','볼로디미르 젤렌스키','30대','긴급체포','프로야구 대구 KS 4차전', '포스트시즌',
            'PS 20G 연속 매진','수도','먼바다','삼성 vs 기아','KS 4차전','삼성라이온즈파크','2만3천550석','KS 3차전','9회초','제네시스 챔피언십','아스트라제네카',
            '전 임원','불법활동','블룸버그 통신','최고상업책임자','에바 인']


custom_dict = pd.DataFrame({"word":word_list})
custom_dict['morp'] = "NNP"

custom_dict.to_csv(r"C:\Users\user\Desktop\SB\ZeroTo\custom_dict.txt", index = False, header = False, sep="\t")


# Naver 뉴스 API 호출 함수
def get_news():
  encText = urllib.parse.quote("뉴스")
  url = "https://openapi.naver.com/v1/search/news?query=" + encText + "&display=100" +"&sort=date" +"&start=1" # JSON 결과
# url = "https://openapi.naver.com/v1/search/news.xml?query=" + encText # XML 결과

  request = urllib.request.Request(url)
  request.add_header("X-Naver-Client-Id",client_id)
  request.add_header("X-Naver-Client-Secret",client_secret)
  response = urllib.request.urlopen(request)
  rescode = response.getcode()

  if(rescode==200):
    response_body = response.read()
    json_data = json.loads(response_body.decode('utf-8'))
    news_items = json_data['items']
    df = pd.DataFrame(news_items)
    df = df[['title', 'link', 'description', 'pubDate']]
    df.to_csv('news_data.csv', index=False, encoding='utf-8')
    news_data = pd.read_csv('news_data.csv')
    return news_data # 뉴스 기사 리스트 반환
  else:
    print(f"Error Code: {rescode}")
    return None

# 텍스트 정제 함수
pattern0 = r'[\\\-\+<>\/,.""\(\)·©=@\[\]&;\'‘’“…|:ⓒ`※”▲ㆍ?_∼◇△↓\u00A0\u2009\u202F\u2800]|quot'


def clean_text(text):
  if isinstance(text, str):
    return re.sub(pattern0,' ', text).strip()
  else:
    return ''

# 전처리 함수 (예: 텍스트 정제 및 형태소 분석)
def preprocess_text(news_data):
    komoran.set_user_dic(r"C:\Users\user\Desktop\SB\ZeroTo\custom_dict.txt")  # 사용자 정의 사전 설정
    #정규화할 단어들
    dic_standardization = {'유커':'요우커','KS':'한국시리즈','포스트 시즌':'포스트시즌','PS':'포스트시즌'}

    Stopwords = ['화면','라인','뉴스','사진','노컷','기자','제공','조이','화제','이번','제보','앞','포토','연합','각종','말','위크','나이스','영상','아주','룸',
                 '김명준']
    processed_docs = []

    news_data['title'] = news_data['title'].progress_apply(clean_text)
    news_data['description'] = news_data['description'].progress_apply(clean_text)

    for _, news in news_data.iterrows():
            # 제목과 내용을 결합하여 키워드 생성
            keywords = news['title'].strip() + ' ' + news['description'].strip()

            # 명사 추출
            tokens = komoran.nouns(keywords)

            # 정규화 작업 (정규화 사전에 있는 단어만 변환)
            tokens = [dic_standardization.get(w, w) for w in tokens]


            # Stopwords 제거
            tokens = [w for w in tokens if w not in Stopwords]
            processed_docs.append(tokens)

    return processed_docs

# LDA model function
def lda(text, k_model, iteration, word_remove=0):
    model = tp.LDAModel(k=k_model, rm_top=word_remove, seed=871017,alpha = 0.02, eta = 0.1)

    for line in text:
        model.add_doc(line)

    model.burn_in = 150
    model.train(0)
    model.optim_interval =5




    for i in range(0, iteration, 10):
        model.train(100)


    model.summary()
    return model

def run_lda():
    # LDA 분석 실행
    news_data = get_news()

    if news_data is not None:
        print("뉴스를 불러왔습니다. 전처리 및 LDA 모델링을 시작합니다.")

        # 새 뉴스 데이터를 DataFrame에 추가
        news_data = pd.DataFrame(news_data)
        processed_docs = preprocess_text(news_data)

        # 새로운 LDA 모델을 생성 (k는 토픽 수)
        model = tp.LDAModel(k=10)
        for doc in processed_docs:
            model.add_doc(doc)
        model.train(1000)  # 모델 훈련

        topic_wf_df = pd.DataFrame()
        for i in range(10):  # Adjust based on the number of topics
              temp = pd.DataFrame(model.get_topic_words(i, top_n=10))
              temp.columns = [f"Topic{i}", f"probs{i}"]
              temp = temp.reset_index()

              if i == 0:
                  topic_wf_df = pd.concat([topic_wf_df, temp], ignore_index=True)
              else:
                  topic_wf_df = topic_wf_df.merge(temp, left_on="index", right_on="index")

            # Document-topic distribution
        theta_df = pd.DataFrame()
        for line in model.docs:
            temp = pd.DataFrame(line.get_topic_dist())
            theta_df = pd.concat([theta_df, temp.T])

            # Clean up theta_df
        theta_df.columns = [f'Topic{x}' for x in range(10)]
        theta_df['Highest_Topic'] = theta_df.idxmax(axis=1)

                        # Print topic results
        topic_results = {}
        topic_counts = theta_df['Highest_Topic'].value_counts()
        selected_words = set()  # 중복 방지를 위한 선택된 키워드 집합
        
        for i, topic in enumerate(topic_counts.index):
            topic_num = int(topic.replace("Topic", ""))
            topic_words = model.get_topic_words(topic_num, top_n=10)  # top_n=10개의 단어 가져오기
            
            # 중복이 없으면 첫 번째 단어 선택, 중복 시 대체 단어 선택
            for topic_word, _ in topic_words:
                if topic_word not in selected_words:
                    topic_results[f'TOPIC #{i} WORD'] = topic_word
                    selected_words.add(topic_word)  # 선택된 단어 기록
                    print(f'TOPIC #{i} WORD: {topic_word}')
                    break
            else:
                # 모든 키워드가 중복되었을 경우
                topic_results[f'TOPIC #{i} WORD'] = None
                print(f'TOPIC #{i} WORD: None')
        
        print(topic_results)

        
        
        requests.post("http://172.30.1.92:8000/update_results/", json={"results": topic_results})
        


                    

if __name__ == "__main__":
    start_time = time.time()
    while time.time() - start_time < 300:
        try:
            run_lda()  # LDA 분석 실행 및 결과 전송
            time.sleep(60)  # 다음 사이클 대기
        except Exception as e:
            print(f"오류 발생: {e}")
            break  # 오류 발생 시 루프 종료




