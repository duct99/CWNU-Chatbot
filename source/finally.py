import streamlit as st
from langchain.chat_models import ChatOpenAI
import os
from PIL import Image
import pandas as pd
import time
import random

os.environ["OPENAI_API_KEY"] = "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"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path=r"C:\Users\kimsoo\Desktop\source\Database.csv", encoding='utf-8-sig')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
texts1 = text_splitter.split_documents(documents)

loader = CSVLoader(file_path=r"C:\Users\kimsoo\Desktop\source\Database_T.csv", encoding='utf-8-sig')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
texts2 = text_splitter.split_documents(documents)

loader = CSVLoader(file_path=r"C:\Users\kimsoo\Desktop\source\Database_description.csv", encoding='utf-8-sig')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
texts3 = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(texts1+texts2+texts3, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template="""
### Guidelines:
- 답변 시에는 최대한 친절하고 친근한 말투를 사용하여야 합니다.
- 사용자가 질문한 내용에 대해서 최대한 정확하고 간결하게 답변하여야 합니다.
- 사용자가 요구하는 모든 질문에 대해서 누락이나 생략 없이 모두 답변하여야 합니다.
- 주어진 정보로 알 수 없는 내용이 답변에 포함되어서는 안됩니다.

----------------
{summaries}

답변은 기본적으로 한국어와 마크다운 포맷으로 이뤄져야 합니다:
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

chain_type_kwargs = {"prompt": prompt}

llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=1)  # Modify model_name if you have access to GPT-4

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

def generate_response(prompt, chat_history):
    full_prompt = "\n".join(chat_history + [prompt])
    result = chain(full_prompt)
    return result['answer']

def name():
    program_name = pd.read_csv('Database.csv').loc[:, '부제목'].to_list()
    result = '개설된 프로그램명 나열해 드릴게요'
    return result, program_name

def eng():
    eng_program = pd.read_csv('Database.csv')
    eng_filtered = eng_program[eng_program['부제목'].str.contains('토익|토스|오픽')]
    
    eng_name_list = eng_filtered['부제목'].tolist()
    eng_date_list = eng_filtered['마감여부'].tolist()
    
    result = '어학 프로그램 신청 마감기한에 대해 알려드릴게요'
    return [result, eng_name_list, eng_date_list]

def mi():
    mi_pd = pd.read_csv('Database.csv')
    mi_top = mi_pd.sort_values(by='마일리지', ascending=False).head()
    
    mi_top_name = mi_top['부제목'].tolist()
    mi_top_mi = mi_top['마일리지'].tolist()
    
    result = '마일리지 상위5개 프로그램을 나열해 드릴게요'
    return [result, mi_top_name, mi_top_mi]

st.markdown('<br><br><br><br><br><br><br><br><br>', unsafe_allow_html=True)

    
# 이미지 로드 및 크기 조정
image_path = "C:\\Users\\kimsoo\\Desktop\\source\\blue.PNG"
image = Image.open(image_path)
original_width, original_height = image.size
new_height = 250
new_width = int(new_height * (original_width / original_height))
image = image.resize((new_width, new_height))

# 중앙에 이미지 배치
col1, col2, col3 = st.columns([4, 4, 4])
with col2:
    st.image(image, width=200, use_column_width='auto')

st.markdown('<br>', unsafe_allow_html=True)

# 페이지 하단에 빈 공간 추가
for _ in range(10):  # 필요한 만큼의 빈 줄 수 조정
    st.empty()
    
# 아이콘 로드 및 배치
icon_path = "C:\\Users\\kimsoo\\Desktop\\source\\icon.png"
icon = Image.open(icon_path)
col1, col2, col3, col4 = st.columns([3, 1, 8, 3])
with col2:
    st.empty()

# 추가 텍스트
with col3:
    st.markdown("**저는 창원대학교 비교과 알리미 청심황입니다!**")
    st.markdown("**⠀⠀⠀⠀⠀⠀⠀⠀무엇을 도와드릴까요?**")


# 페이지 하단에 빈 공간 추가
for _ in range(10):  # 필요한 만큼의 빈 줄 수 조정
    st.empty()

# st.session_state.messages를 사용하는 나머지 부분
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 예를 들어, 메시지를 반복해서 표시
for msg in st.session_state.messages:
    # 메시지를 표시하는 코드
    pass

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

E_dream_image = Image.open(r"C:\Users\kimsoo\Desktop\source\edream.png")
original_width, original_height = E_dream_image.size
new_width = 250
new_height = int(new_width * (original_height / original_width))
E_dream_image = E_dream_image.resize((new_width, new_height))

Waggle = Image.open(r"C:\Users\kimsoo\Desktop\source\waggle.png")
original_width, original_height = Waggle.size
new_height = int(new_width * (original_height / original_width))
Waggle = Waggle.resize((new_width, new_height))

freq = Image.open(r"C:\Users\kimsoo\Desktop\source\freq.png")
original_width, original_height = freq.size
new_height = int(new_width * (original_height / original_width))
freq = freq.resize((new_width, new_height))

E_dream_image.save(r"C:\Users\kimsoo\Desktop\source\edream.png")
Waggle.save(r"C:\Users\kimsoo\Desktop\source\waggle.png")

with st.sidebar:
    st.sidebar.image(E_dream_image, use_column_width=False)
    st.markdown("[이뤄드림 바로가기](https://edream.changwon.ac.kr/)")
    st.sidebar.image(Waggle, use_column_width=False)
    st.markdown("[와글 바로가기](https://www.changwon.ac.kr/portal/main.do)")
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.sidebar.image(freq, use_column_width=False)
    st.write('')
    st.write('')
    
    if st.button('개설 프로그램 확인'):
        result, name_list = name()
        name_info = ''
        for i, j in enumerate(name_list):
            name_info += f"{i+1}. {j}\n\n"
        st.session_state.output_text = f"{result}\n\n{name_info}"
        
    if st.button('어학 프로그램 신청 마감기한'):
        result, eng_name_list, eng_date_list = eng()
    
        eng_info = ""
        for i, (program_name, deadline) in enumerate(zip(eng_name_list, eng_date_list)):
            eng_info += f"{i+1}. 프로그램명: {program_name}, \t 신청 마감기간: {deadline}\n\n"

        st.session_state.output_text = f"{result}\n\n{eng_info}"
        
    if st.button('마일리지 높은 프로그램'):
        result, mi_top_name, mi_top_mi = mi()
        
        mi_info = ""
        for i, (program_name, program_mi) in enumerate(zip(mi_top_name, mi_top_mi)):
            mi_info += f"{i+1}. 프로그램명: {program_name}, \t 마일리지: {program_mi}\n\n"
        
        st.session_state.output_text = f"{result}\n\n{mi_info}"
        
if 'output_text' in st.session_state:
    st.write(st.session_state.output_text)
        
if prompt := st.chat_input():
    print(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    loading_messages = [
    "창원대학교의 주소는 '경상남도 창원시 의창구 창원대학로 20'입니다.",
    "창원대학교는 대한민국 경상남도 창원시에 위치한 공립 고등교육기관입니다.",
    "창원대학교 뒷편에 위치한 창원중앙역의 부기역명은 창원대역입니다.",
    "창원대학교 캠퍼스 내부에는 '산애산'이라는 막걸리집이 있습니다.",
    "2023년 12월 21일은 캡스톤 디자인 결과 발표일이자, 종강일입니다."
]
    random_loading_message = random.choice(loading_messages)
    
    # 로딩 스피너 표시
    with st.spinner(f'{random_loading_message}'):
        # 처리 시간을 모방하기 위한 딜레이
        time.sleep(2)
        chat_history = [msg["content"] for msg in st.session_state.messages if msg["role"] != "assistant"]
        assistant_msg = generate_response(prompt, chat_history)
        st.balloons()
        
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").write(assistant_msg)