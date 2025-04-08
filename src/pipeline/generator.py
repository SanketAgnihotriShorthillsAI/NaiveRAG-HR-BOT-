import os
import logging
from dotenv import load_dotenv
import httpx
import asyncio

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Generator:
    def __init__(
        self,
        use_openai=False,
        use_optimize_llm=False,
        gemini_model="gemini-1.5-flash-latest",
        openai_model="gpt-3.5-turbo",
    ):
        self.use_openai = use_openai
        self.use_optimize_llm = use_optimize_llm
        self.gemini_model_name = gemini_model
        self.openai_model_name = openai_model

        if self.use_optimize_llm:
            logging.info(f"🔌 Using OptimizeRag Azure OpenAI deployment: {AZURE_OPENAI_DEPLOYMENT}")
        elif use_openai:
            import openai
            openai.api_key = OPENAI_API_KEY
            self.openai = openai
            logging.info(f"🔌 Using OpenAI: {openai_model}")
        else:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.genai = genai
            self.model = genai.GenerativeModel(gemini_model)
            logging.info(f"🔌 Using Gemini: {gemini_model}")

    async def generate_response(self, query, retrieved_chunks):
        return await self._generate(query, retrieved_chunks)


    async def _generate(self, query, retrieved_chunks):
        logging.info(f"🧠 Generating response for query: {query}")
        if not retrieved_chunks:
            return "⚠️ No relevant information found.", "⚠️ No relevant information found."

        context = "\n\n".join([f"- {chunk}" for chunk in retrieved_chunks])

        prompt = f"""---Role---

You are an intelligent and efficient HR assistant that helps analyze and respond to queries related to resumes and candidate information provided in the Knowledge Base.

---Goal---

Generate a clear, helpful, and concise response using only the resume data available in the Knowledge Base. This may include details such as education, experience, skills, certifications, projects, achievements, role transitions, work preferences, availability or listing all eligible candidates accroding to the conditions in query.

Tailor your response to the intent behind the HR query, which may involve:
- Identifying suitable candidates for specific technologies, tools, roles, or industries.
- Filtering based on attributes like location, experience level, education, certifications, gaps, or functional transitions.
- Summarizing key skills or project work for a given candidate or group.
- Recognizing patterns such as career shifts, sabbaticals, leadership roles, or domain expertise.
- **Insight-oriented**: Help HR understand patterns like shifts in roles, popular technologies in recent roles, commonly held degrees, or industry transitions.

---Knowledge Base---
{context}

---User Query---
{query}

---Response Rules---
- Target the response length and format according to the query.
- Candidate names should be clearly identified and, if available, extracted from metadata or the resume text; minor formatting differences (e.g., extra characters, file extensions) should be ignored.
- Do not assume anything that is not explicitly supported by the resume data.
- Use markdown formatting (headings, bullet points, or tables) where it enhances clarity.
- Redact or omit any critical personal information (phone numbers, email addresses, DOB, etc.).
- Maintain a professional, concise tone and answer the query directly and accurately.
- Respond in the same language as the query.
- If you do not know the answer, state so clearly.
- Avoid inferring candidate intent, strengths, or preferences unless explicitly mentioned.
- Ensure that the answer is well-grounded by strictly using the provided Knowledge Base.
"""

        try:
            if self.use_optimize_llm:
                print("Using OptimizeRag Azure LLM...")
                result = await self._generate_optimize_llm(query, prompt)
            elif self.use_openai:
                messages = [
                    {"role": "system", "content": "You are an HR assistant trained on candidate resumes."},
                    {"role": "user", "content": prompt}
                ]
                response = self.openai.ChatCompletion.create(
                    model=self.openai_model_name,
                    messages=messages,
                    temperature=0.3,
                )
                result = response.choices[0].message.content.strip()
            else:
                response = self.model.generate_content(prompt)
                result = response.text.strip()
            
            append_naive_log(query, context, result)

            return result, result

        except Exception as e:
            logging.error(f"❌ Generation error: {e}")
            return "⚠️ Error generating response.", "⚠️ Error generating response."

    async def _generate_optimize_llm(self, user_prompt, system_prompt=""):
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        }

        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 6144,
        }

        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(url, headers=headers, json=body)
            res.raise_for_status()
            response = res.json()
            return response["choices"][0]["message"]["content"]

import json

def append_naive_log(query, context, response, log_path="naiveRag_query_logs.json"):
    entry = {
        "query": query,
        "context": context,
        "response": response
    }

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            json.dump([], f)

    with open(log_path, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)
