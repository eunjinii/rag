from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt-only LLM invocation
def run_simple_prompt(model, question, options=None):
    if options is None:
        template = '''You are an AI assistant for grad students.
        Answer the following question in concise sentences.
        Question: {question}'''
    else:
        template = '''You are an AI assistant for grad students.
        Answer the question and select the most appropriate answer from the given choices.
        You must return only the correct answer (a, b, c or d) and nothing else.
        
        Question: {question}
        Choices:
        {options}

        You must return a single character (a, b, c or d). Do not provide any explanation.'''
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = (prompt | model | StrOutputParser())
    answer = chain.invoke({"question": question, "options": options}, config={"timeout": 30}).strip()
    return answer

# RAG-enabled LLM invocation
def run_rag_chain(model, retriever, question, options=None, k=2, is_rerank=False):
    def format_docs(docs):
        if hasattr(docs[0], 'page_content'):
            return '\n\n'.join([d.page_content for d in docs])
        else:
            return '\n\n'.join([d[0] for d in docs])
    
    if options is None:
        template = '''Here are the relevant documents: {context}
        
        You are an AI assistant for graduate students.
        Use the provided context to answer the question.
        Answer the following question in concise sentences.

        Question: {question}

        Output only the answer. Do not provide any explanation.'''
    else:
        template = '''Here are the relevant documents: {context}
        
        You are an AI assistant for graduate students.
        Use the provided context to select the most accurate answer.
        Return only one letter (a, b, c, or d) with no additional text or explanation.

        Question: {question}
        Potential choices: {options}

        Output only one character (a, b, c, or d) with no explanation.'''

    prompt = ChatPromptTemplate.from_template(template)
    retrieved_docs = retriever.search(query=question, k=k, rerank=is_rerank)
    formatted_context = format_docs(retrieved_docs)
    chain = (prompt | model | StrOutputParser())
    answer = chain.invoke({"context": formatted_context, "question": question, "options": options}, config={"timeout": 30}).strip()
    return answer
