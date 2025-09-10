import os
import re
import pickle
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked
)
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# ---------------- Load ENV ----------------
load_dotenv()

# HuggingFace Free Model
Free_model_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
Free_model = ChatHuggingFace(llm=Free_model_endpoint)


# ---------------- Utility: Extract Video ID ----------------
def get_video_id_regex(url: str) -> str | None:
    """Extract YouTube video ID from normal and short links."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


# ---------------- Transcript Fetching ----------------
def fetch(video_url: str):
    """Fetch transcript text from YouTube video with proxy support."""
    video_id = get_video_id_regex(video_url)
    if not video_id:
        st.error("Invalid YouTube URL — couldn't extract video ID.")
        return None

    # ✅ Load proxy from environment variables
    proxies = {
        "http": os.getenv("HTTP_PROXY"),
        "https": os.getenv("HTTPS_PROXY"),
    }

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
        text = " ".join([t["text"] for t in transcript])
        return text
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        st.error("Transcript not available for this video.")
        return None
    except RequestBlocked:
        st.error("Transcript request was blocked by YouTube. Try changing proxy or wait.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


# ---------------- Vector Store Creation ----------------
def create_vector_store(text: str, model_type: str, openai_key: str = None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    if model_type == "paid":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def get_llm(model_type, openai_key=None):
    if model_type == "paid":
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_key)
    else:
        return Free_model


# ----------------- Streamlit Styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://m.gettywallpapers.com/wp-content/uploads/2022/05/White-Aesthetic-Background-Photos.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
     /* Header (navbar at the top) */
    header[data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0.6); /* Black with translucency */
    }
    header[data-testid="stHeader"]::before {
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- Streamlit UI ----------------
st.title("📽️ YouTube Chat Bot ")
st.write("Choose between **Paid (OpenAI)** or **Free (HuggingFace)** models.")

model_choice = st.radio("Select Model Type:", ("Paid (OpenAI)", "Free (HuggingFace)"))

if model_choice == "Paid (OpenAI)":
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
else:
    openai_api_key = None

video_url = st.text_input("Enter YouTube Video URL:")

# -------- Process Transcript --------
if st.button("Process Transcript"):
    if not video_url.strip():
        st.error("⚠️ Please enter a valid YouTube video URL.")
    elif model_choice == "Paid (OpenAI)" and not openai_api_key.strip():
        st.error("⚠️ Please enter your OpenAI API key for Paid model.")
    else:
        with st.spinner("Fetching transcript..."):
            transcript_text = fetch(video_url)

        if transcript_text:
            st.session_state.transcript = transcript_text
            with st.spinner("Creating vector store..."):
                model_type = "paid" if model_choice == "Paid (OpenAI)" else "free"
                vector_store = create_vector_store(transcript_text, model_type, openai_key=openai_api_key)

            st.session_state.vector_store = vector_store
            st.success("✅ Transcript processed and stored in memory!")
            st.write(f"Number of chunks created: {len(vector_store.index_to_docstore_id)}")

# -------- Show Transcript --------
if "transcript" in st.session_state:
    st.subheader("📜 Transcript Preview")
    st.text_area("Transcript", st.session_state.transcript[:3000], height=200)

# -------- Chat Section --------
st.subheader("💬 Chat with the Video")
query = st.text_input("Ask a question about the video:")

if st.button("Get Answer"):
    if "vector_store" not in st.session_state:
        st.error("⚠️ No transcript processed yet. Please process one first.")
    elif not query.strip():
        st.error("⚠️ Please enter a question.")
    else:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        llm = get_llm("paid" if model_choice == "Paid (OpenAI)" else "free", openai_key=openai_api_key)

        # Prompt Template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an assistant that answers questions based on the provided transcript.\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )

        # Runnable Chain
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | llm
        )

        with st.spinner("Generating answer..."):
            response = chain.invoke(query)

        st.markdown("**Answer:**")
        st.write(getattr(response, "content", str(response)))
