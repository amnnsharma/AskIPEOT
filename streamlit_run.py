__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from typing import Set
#from openai_gpt_faiss import run_llm
from openai_gpt_chroma import run_llm
import streamlit as st
from streamlit_chat import message

try:
    with st.container():
        st.header("Ask IPEOT 🤖")
    
    if (
        "chat_answers_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
    ):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []


    prompt = st.text_input("Ask your question here:", placeholder="Enter your message here...") or st.button(
        "Submit"
    )

    if prompt:
        with st.spinner("Generating response..."):
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )

            formatted_response = f"{generated_response}"

            st.session_state.chat_history.append((prompt, generated_response))
            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(formatted_response)

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message(
                user_query,
                is_user=True,
            )
            message(generated_response)
except Exception as e:
    st.write("Wait for some time! Too many requests :)")
    st.write(e)
