import streamlit as st
import weaviate
import torch
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer
from together import Together
from difflib import SequenceMatcher
import traceback
import logging
from weaviate.classes.init import Auth
import re

# === Konfigurasi Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatbot_log.txt"), logging.StreamHandler()]
)

# === Konfigurasi Halaman Streamlit ===
st.set_page_config(page_title="Chatbot Hukum Pidana", layout="centered")
logging.info("Streamlit halaman dikonfigurasi")

# === API Keys dan Konfigurasi ===
TOGETHER_API_KEY = "a7f9dbd03514a193189180c5b08f617f2bf540be1babfc2c62070812a69cedcf"
weaviate_url = "fzvdjpkntokay1shmgqcw.c0.asia-southeast1.gcp.weaviate.cloud"
weaviate_api_key = "GhPVi2jpLt2Is9zsidNl6aW6MyBRJk24APLa"

# === Koneksi ke Weaviate ===
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    assert client.is_ready(), "Weaviate tidak siap!"
    logging.info("Berhasil terhubung ke Weaviate")
except Exception as e:
    logging.error("Gagal koneksi ke Weaviate: " + traceback.format_exc())
    st.stop()

collection = client.collections.get("Hukum")

# === Load model Jina lokal ===
@st.cache_resource
def load_jina_model():
    logging.info("Memuat model Jina lokal...")
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    logging.info("Model Jina berhasil dimuat")
    return model, tokenizer

model, tokenizer = load_jina_model()

# === Embedding lokal dengan Jina ===
def get_jina_embedding_local(content):
    logging.info(f"Menghitung embedding untuk: {content[:60]}...")
    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy().tolist()

# === Setup LLM ===
llm_client = Together(api_key=TOGETHER_API_KEY)

def call_llm(messages, max_tokens=512, temperature=0.7, top_p=0.95):
    logging.info("Menghubungi Together AI...")
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        logging.info("Jawaban LLM berhasil diterima")
        return response.choices[0].message.content.strip()
    except Exception:
        logging.error("Error saat memanggil Together AI:\n" + traceback.format_exc())
        st.error("Error saat menghubungi model AI.")
        return ""

# === Hybrid Retrieval ===
def hybrid_retrieval(query, top_k=5):
    try:
        logging.info(f"Melakukan hybrid retrieval untuk query: {query}")
        vector = get_jina_embedding_local(query)

        response = collection.query.hybrid(
            query=query,
            alpha=0.5,
            vector=vector,
            limit=top_k * 2,
            return_properties=["content"]
        )

        results = response.objects  # FIX: akses langsung, bukan .data

        if not results:
            logging.info("Tidak ada hasil dari hybrid retrieval.")
            return []

        scored = []
        query_vec = torch.tensor([vector])
        for obj in results:
            content = getattr(obj, "content", "") or obj.properties.get("content", "")
            content_vec = torch.tensor(get_jina_embedding_local(content))
            score = util.cos_sim(query_vec, content_vec).item()
            scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        logging.info(f"Hybrid retrieval selesai, ditemukan {len(scored)} hasil")
        return [content for score, content in scored[:top_k]]

    except Exception:
        logging.error("Hybrid Retrieval Error: " + traceback.format_exc())
        return []

def knowledge_graph_lookup(query, top_k=2):
    try:
        logging.info("Melakukan KG lookup...")
        vector = get_jina_embedding_local(query)
        collection = client.collections.get("Hukum")
        response = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_properties=["content"]
        )
        return [getattr(obj, "content", "") for obj in response.objects]
    except Exception:
        logging.warning("KG lookup error:\n" + traceback.format_exc())
        return []


def adaptive_top_k(query, min_k=3, max_k=10):
    length = len(query.split())
    return min_k if length <= 5 else max_k if length >= 20 else min_k + (length - 5) * (max_k - min_k) // 15

def merge_concontents(kg, hybrid):
    logging.info("Menggabungkan konteks KG dan Hybrid")
    return "\n\n".join(dict.fromkeys(kg + hybrid))

def evaluate_answer(concontent, answer, threshold=0.1):
    try:
        logging.info("Mengevaluasi relevansi jawaban dengan konteks...")
        ctx_vec = get_jina_embedding_local(concontent)
        ans_vec = get_jina_embedding_local(answer)
        score = util.cos_sim(torch.tensor([ctx_vec]), torch.tensor([ans_vec])).item()
        logging.info(f"Similarity score: {score}")
        return score >= threshold
    except Exception:
        logging.warning("Evaluasi relevansi gagal:\n" + traceback.format_exc())
        return False

# === Pipeline RAG ===
def rag_chat(query, system_prompt, max_tokens, temperature, top_p):
    query = query.strip()
    if not query:
        return "Silakan ketik pertanyaan terlebih dahulu."

    top_k = adaptive_top_k(query)
    hybrid_ctx = hybrid_retrieval(query, top_k)
    kg_ctx = knowledge_graph_lookup(query, top_k=2)
    concontent = merge_concontents(kg_ctx, hybrid_ctx)
    print(concontent)
    if not concontent:
        return "Maaf, tidak ditemukan konteks hukum relevan."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"### Konteks:\n{concontent}\n\n### Pertanyaan:\n{query}"}
    ]
    answer = call_llm(messages, max_tokens, temperature, top_p)

    if not answer:
        return "Model AI tidak memberikan jawaban."

    similarity = SequenceMatcher(None, query.lower(), answer.lower()).ratio()
    if similarity > 0.9:
        logging.info("Jawaban terlalu mirip dengan pertanyaan, mungkin tidak informatif.")
        return "Maaf, saya belum bisa memberikan jawaban yang informatif."

    if not evaluate_answer(concontent, answer):
        logging.info("Jawaban mungkin kurang relevan.")
        answer = f"Jawaban mungkin kurang relevan.\n\n{answer}"

    return answer

def extract_answer(text):
    """Hapus bagian <think>...</think> dari jawaban"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# === Streamlit UI ===
st.title("🔍 Chatbot Konsultasi Hukum Pidana 🇮🇩")
st.write("Tanyakan apa pun tentang KUHP, KUHAP, dan UU Pidana Khusus. Jawaban berdasarkan referensi hukum Indonesia.")

with st.sidebar:
    st.header("⚙️ Pengaturan LLM")
    system_msg = st.text_area("System Message", value="Kamu adalah asisten hukum pidana yang cerdas dan dapat memahami Bahasa Indonesia. Jawablah pertanyaan hukum berdasarkan konteks dokumen hukum Indonesia yang diberikan, dengan bahasa yang jelas, sopan, dan mudah dimengerti oleh orang awam.")
    max_tokens = st.slider("Max Tokens", 64, 2048, 512)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95)

query = st.text_input("Masukkan pertanyaan hukum pidana Anda:")
if st.button("Tanyakan"):
    logging.info(f"Pengguna bertanya: {query}")
    with st.spinner("Mencari jawaban..."):
        response = rag_chat(query, system_msg, max_tokens, temperature, top_p)
    final_answer = extract_answer(response)
    st.markdown("### Jawaban:")
    st.write(final_answer)
    logging.info(f"Jawaban diberikan: {final_answer[:80]}...")
client.close()
