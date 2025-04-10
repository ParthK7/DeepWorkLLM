from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "gemma3:1b")

template = """
You are an expert in the concept of Deep work 

Here are some relevant tips : {tips}

Here is a question to answer : {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

while True:
    print("\n\n ==============================")
    question = input("Ask your question ('/bye' to quit)")
    print("\n\n")
    if question == "/bye":
        break

    answer = retriever.invoke(question)
    tips = "\n\n.join([doc.page_content for doc in answer])"
    result = chain.invoke({"tips" : tips, "question" : question})
    print(result)