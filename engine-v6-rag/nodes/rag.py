from utilities.states.input_state import ChatState
from service.rag.load_document import load_doctor_documents
from service.rag.langchain_document import to_langchain_documents
from service.rag.vector_store import create_faiss_vectorstore
from service.rag.search.hybrid import hybrid_langchain_retriever
from utilities.llms.chat_llm import chat_llm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
import os

def rag_node(state: ChatState):
    print("rag node executed")
    # get user input from chatbot
    user_input = state["messages"][-1].content
    print("user input: ", user_input)

    # extract the name of the doctor or department from the user input using llm
    llm_prompt_nd = f"""
You are an expert medical entity extractor.

Your task:
- Extract ONLY the doctor's name OR the department name from the user input.
- The user may write in any format, any language, informal or formal.
- The input may contain extra words like booking, appointment, dekhaite chai, amar lagbe, please, etc. IGNORE them.
- If a PERSON name appears, assume it is a DOCTOR NAME.
- If a medical specialty appears, assume it is a DEPARTMENT.

STRICT RULES:
- Output MUST be ONLY ONE of the following:
  1) Doctor name (e.g. "Dr. Md. Rakibul Hossain" or "Rakibul Hossain" or "rakibul")
  2) Department name (e.g. "Cardiology", "Orthopedics")
  3) The word: Unknown

- Do NOT explain.
- Do NOT add punctuation.
- Do NOT add extra words.
- Try VERY HARD before returning "Unknown".
- Return "Unknown" ONLY if there is absolutely no doctor name or department.

Examples:
Input: "I want to book an appointment with Dr Rakibul"
Output: Dr Rakibul

Input: "heart doctor dekhaite chai"
Output: Cardiology

Input: "orthopedic er doctor lagbe"
Output: Orthopedics

Input: "ami ekta appointment nite chai"
Output: Unknown

User input:
{user_input}"""

    llm_response_nd = chat_llm.invoke([HumanMessage(content=llm_prompt_nd)])

    # check if the llm response is unknown
    if llm_response_nd.content == "Unknown":
        print("llm response: ", llm_response_nd.content)
        return {
            "messages": ["I'm sorry, I didn't get the name of the doctor or department. Please tell me the name of the doctor or department again."],
            "track_stage": "rag"
        }
    
    else:
        extracted_name = llm_response_nd.content
        

    # todo: rag things here
    # load the doctors document from the local system
    documents = load_doctor_documents("documents/doctors.json")

    # convert the raw documents to langchain documents
    langchain_documents = to_langchain_documents(documents)
    # print("langchain_documents: ", langchain_documents)
    
    
    # load the vectorstore
    vectorstore = FAISS.load_local(
        "vectorstore/db_faiss", 
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
        )

    # get the similar documents using hybrid retriever
    result = hybrid_langchain_retriever(
        query=extracted_name,
        lc_documents=langchain_documents,
        vectorstore=vectorstore
    )

    print(f"retrieved documents: {result}")

    # llm prompt
    llm_prompt_doctor_verification = f"""
    You are a medical appointment booking chatbot. You are given a doctor name or department name and a list of documents (all related doctor name and department name, etc.). The user input is: {user_input}. The user can say the name of the doctor or the department of the doctor. We have the following doctors and departments in our hospital: {langchain_documents}. Verify and Tell user about the doctor or the department he is asking for. If our hospital does not have the doctor or the department he is asking for, say that we do not have the doctor or the department he is asking for.
"""
    llm_response = chat_llm.invoke([HumanMessage(content=llm_prompt_doctor_verification)])

    doctor_verification_result = llm_response.content

    # show sample message
    # sample_message = "I got your message. I am a rag node. I will do rag things here. tell me your name"
    # print("sample message: ", sample_message)
    # return the message
    return {
        "messages": [doctor_verification_result],
        "track_stage": "1"
    }