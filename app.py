import streamlit as st
import pandas as pd
import json
import re
import os
import google.generativeai as genai


# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="Granite AI Chat Advisor ", page_icon="🪨")

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"


# ==========================================================
# DATA
# ==========================================================

@st.cache_data
def load_data():
    file_path = os.path.join(os.getcwd(), "granite_master_dataset.csv")
    return pd.read_csv(file_path, encoding="utf-8-sig")

df = load_data()

# ==========================================================
# UTIL
# ==========================================================
# ==========================================================
# SESSION STATE INIT (ต้องมี)
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

def extract_budget(text):
    match = re.search(r"งบ\s*([\d,]+)", text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def smart_filter(df, user_input, budget):

    filtered = df.copy()

    # 1. Budget
    if budget:
        filtered = filtered[filtered["price_min"] <= budget]

    # 2. Indoor / Outdoor (only if user mentions)
    if "นอก" in user_input or "outdoor" in user_input:
        filtered = filtered[
            filtered["indoor_outdoor"].str.contains("outdoor", case=False)
        ]
    elif "ใน" in user_input or "indoor" in user_input:
        filtered = filtered[
            filtered["indoor_outdoor"].str.contains("indoor", case=False)
        ]

    # 3. Usage
    if "ครัว" in user_input or "counter" in user_input:
        filtered = filtered[
            filtered["popular_use"].fillna("").str.contains("countertop", case=False)
        ]
    elif "พื้น" in user_input or "floor" in user_input:
        filtered = filtered[
            filtered["popular_use"].fillna("").str.contains("floor", case=False)
    ]

    elif "ผนัง" in user_input or "wall" in user_input:
        filtered = filtered[
            filtered["popular_use"].fillna("").str.contains("wall", case=False)
    ]

    # 4. Style
    styles = []
    if "minimal" in user_input or "มินิมอล" in user_input:
        styles.append("minimal")
    if "modern" in user_input:
        styles.append("modern")
    if "luxury" in user_input or "หรู" in user_input:
        styles.append("luxury")

    for style in styles:
        filtered = filtered[
            filtered["style_tag"].str.contains(style, case=False)
        ]

    # Remove pre-order
    filtered = filtered[filtered["stock_status"] != "pre_order"]

    return filtered
def extract_pattern_intent(user_input):

    text = user_input.lower()

    intent = {
        "color": None,
        "pattern": None,
        "style": None
    }

    # 🎨 สี
    if "ขาว" in text:
        intent["color"] = "white"
    elif "ดำ" in text:
        intent["color"] = "black"
    elif "เทา" in text:
        intent["color"] = "gray"
    elif "น้ำตาล" in text:
        intent["color"] = "brown"

    # 🌀 ลาย
    if "เรียบ" in text:
        intent["pattern"] = "solid"
    elif "ลายเส้น" in text or "ไหล" in text:
        intent["pattern"] = "veined"
    elif "จุด" in text or "ประกาย" in text:
        intent["pattern"] = "speckled"

    # ✨ สไตล์
    if "หรู" in text:
        intent["style"] = "luxury"
    elif "มินิมอล" in text:
        intent["style"] = "minimal"
    elif "modern" in text:
        intent["style"] = "modern"

    return intent


def ranking_score(df, budget, user_input):

    df = df.copy()
    df["score"] = 0

    # 1️⃣ Budget proximity (40%)
    if budget is not None:
        max_price = df["price_min"].max()
        df["budget_score"] = 1 - (abs(df["price_min"] - budget) / (max_price + 1))
        df["budget_score"] = df["budget_score"].clip(lower=0)
        df["score"] += df["budget_score"] * 0.4



    # 3️⃣ Style match bonus (20%)
    text = user_input.lower()
    style_bonus = 0

    if "luxury" in text or "หรู" in text:
        df["style_bonus"] = df["style_tag"].fillna("").str.contains("luxury", case=False).astype(int)
        df["score"] += df["style_bonus"] * 0.2

    if "minimal" in text or "มินิมอล" in text:
        df["style_bonus"] = df["style_tag"].fillna("").str.contains("minimal", case=False).astype(int)
        df["score"] += df["style_bonus"] * 0.2

    # 4️⃣ Stock priority (20%)
    df["stock_bonus"] = (df["stock_status"] == "in_stock").astype(int)
    df["score"] += df["stock_bonus"] * 0.2

    return df.sort_values("score", ascending=False)



# ==========================================================
# AI


def init_client():
    if not API_KEY:
        return None
    genai.configure(api_key=API_KEY)
    return genai


client = init_client()  # 🔥 ต้องมีบรรทัดนี้




def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None


def ask_ai_advisor(client, user_input, filtered_df):

    if client is None:
        return None

    top_df = filtered_df.head(5)

    prompt = f"""
คุณคือผู้เชี่ยวชาญด้านหินแกรนิต
เลือก stone_name ได้เฉพาะจากรายการที่ให้
ตอบเป็น JSON เท่านั้น

รายการ:
{top_df[["stone_name","price_min","price_max","style_tag"]].to_json(orient="records", force_ascii=False)}

คำถาม:
{user_input}

ตอบ:
{{
    "recommended_stone": "",
    "finish_type": "",
    "reason": "",
    "warnings": ""
}}
"""

    try:
        model = client.GenerativeModel(MODEL_NAME)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2
            }
        )

        return extract_json(response.text)

    except Exception as e:
        print("AI Error:", e)
        return None



def validate_ai_output(ai_json, filtered_df):

    if not ai_json:
        return None

    # กัน key error
    if "recommended_stone" not in ai_json:
        return None

    if filtered_df is None or filtered_df.empty:
        return None

    stone_list = filtered_df["stone_name"].astype(str).tolist()

    if str(ai_json["recommended_stone"]) not in stone_list:
        return None

    stone_row = filtered_df[
        filtered_df["stone_name"].astype(str) == str(ai_json["recommended_stone"])
    ].iloc[0]

    ai_json["price_range"] = f"{stone_row['price_min']} - {stone_row['price_max']} บาท/ตร.ม."

    return ai_json






# ==========================================================
# MAIN CHAT LOOP
# ==========================================================

user_input = st.chat_input("พิมพ์ความต้องการของคุณ...")

if user_input:

    # 1️⃣ Save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Update Budget Memory
    new_budget = extract_budget(user_input)
    if new_budget:
        st.session_state.memory["budget"] = new_budget

    budget = st.session_state.memory.get("budget")

    # 3️⃣ Initial Filter
    filtered_df = smart_filter(df, user_input, budget)

    # 4️⃣ Intent Refinement
    intent = extract_pattern_intent(user_input)

    if intent["color"]:
        filtered_df = filtered_df[
            filtered_df["base_color_en"] == intent["color"]
        ]

    if intent["pattern"]:
        filtered_df = filtered_df[
            filtered_df["pattern_type"] == intent["pattern"]
        ]

    if intent["style"]:
        filtered_df = filtered_df[
            filtered_df["style_tag"].fillna("").str.contains(
                intent["style"], case=False
            )
        ]

    # 5️⃣ Remove pre-order again (safety)
    filtered_df = filtered_df[
        filtered_df["stock_status"] != "pre_order"
    ]

    # 6️⃣ Fallback if empty
    if filtered_df.empty:

        cheapest_df = df.sort_values("price_min")

        if not cheapest_df.empty:
            best_row = cheapest_df.iloc[0]

            response_text = f"""
❌ ไม่พบหินที่ตรงเงื่อนไขในงบประมาณ {budget}

🪨 ตัวเลือกที่ใกล้เคียงที่สุด:
**{best_row['stone_name']}**

💰 ราคา:
{best_row['price_min']} - {best_row['price_max']} บาท/ตร.ม.

💡 แนะนำเพิ่มงบอีกประมาณ
{max(0, best_row['price_min'] - (budget or 0))} บาท
"""
        else:
            response_text = "ไม่พบข้อมูลในระบบ"

    else:

        # 7️⃣ Ranking
        ranked_df = ranking_score(filtered_df, budget, user_input)

        # 8️⃣ AI
        # ==========================================================
# AI
# ==========================================================



        ai_result = ask_ai_advisor(client, user_input, ranked_df)
        ai_result = validate_ai_output(ai_result, ranked_df)

        # 9️⃣ Build Response
        if ai_result:

            response_text = f"""
🪨 **ลายหรือสีหินแกรนิตที่แนะนำ:** {ai_result['recommended_stone']}

✨ **ผิวที่เหมาะสม:** {ai_result['finish_type']}

💬 **เหตุผล:**  
{ai_result['reason']}

⚠️ **ข้อควรระวัง:**  
{ai_result['warnings']}

💰 **ราคาประมาณ:** {ai_result['price_range']}
"""

        else:

            top3 = ranked_df.head(3)
            recommendations = []

            for _, row in top3.iterrows():

                confidence = min(95, round(row["score"] * 100, 1))

                style_raw = str(row.get("style_tag", "ไม่ระบุ"))
                style_clean = ", ".join(
                    [s.capitalize() for s in style_raw.split("|")]
                )

                recommendations.append(f"""
### 🎨 {row.get('stone_name')}

🖤 โทนสี: {row.get('color_tone')}  
🌍 สีหลัก: {str(row.get('base_color_en')).capitalize()}  
🌀 ลักษณะลาย: {row.get('pattern_type').capitalize()}  
✨ สไตล์: {style_clean}  

💰 ราคา: {row.get('price_min')} - {row.get('price_max')} บาท/ตร.ม.
⭐ ความเหมาะสม: {confidence}%
""")

            response_text = "## 🎨 ลายแกรนิตที่เหมาะกับคุณ\n"
            response_text += "\n".join(recommendations)

    # 🔟 Render (จุดเดียวเท่านั้น)
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )




























