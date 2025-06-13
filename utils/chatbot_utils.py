import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_top_similar(user_q, faq_df, embeddings, client, get_embedding_func, k=1):
    q_vec = np.array(get_embedding_func(user_q, client))
    sims = cosine_similarity([q_vec], np.stack(embeddings))[0]
    idxs = sims.argsort()[::-1][:k]
    return faq_df.iloc[idxs]

def generate_response_with_history(user_q, chat_log, ref_q, ref_a, client, max_turns=5):
    system_prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。"
        "以下のFAQを参考に200文字以内で回答してください。\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in reversed(chat_log[-max_turns:]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_q})
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()
