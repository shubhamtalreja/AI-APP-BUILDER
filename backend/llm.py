from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1200,
)

with open("prompts/intent_classifier.txt", encoding="utf-8") as f:
    intent_template = f.read()

intent_prompt = ChatPromptTemplate.from_template(intent_template)
intent_chain = intent_prompt | llm

with open("prompts/react_app.txt", encoding="utf-8") as f:
    app_template = f.read()

app_prompt = ChatPromptTemplate.from_template(app_template)

app_chain = app_prompt | llm