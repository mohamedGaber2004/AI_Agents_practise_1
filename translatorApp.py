import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain_core.messages import HumanMessage , BaseMessage , SystemMessage


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


st.title("Hello, I'm an AI Translator Application!.")


sourceText = st.text_area(
    "Enter the text you want to translate",
    placeholder="type your text here",
    height=200
)


languages = ["Arabic","English","Chinese"]
targetLanguage = st.selectbox(
    "Select the target language:",
    languages 
)

button = st.button(
    "Translate",
    disabled=not sourceText or not targetLanguage
)

if button : 
    prompt = PromptTemplate.from_template(
        """
            translate the following text to {t} : 

            <text>
            {text}
            </text>

            only provide The translated text without any additional information
        """
    )

    excuted_prompt = prompt.invoke({"t" : targetLanguage , "text" : sourceText})
    output = llm.invoke(excuted_prompt)

    st.write(output.content)
