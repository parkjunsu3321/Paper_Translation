from openai import OpenAI


def translate_text(client: OpenAI, text: str) -> str:
    if not text.strip():
        return text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional academic translator specializing in technical and scientific papers. "
                    "Translate the given text into Korean. "
                    "Rules:\n"
                    "- Preserve all technical terms, proper nouns, and abbreviations in their original form (or add Korean in parentheses).\n"
                    "- Maintain the original tone, structure, and paragraph breaks.\n"
                    "- Do NOT add explanations, summaries, or commentary.\n"
                    "- Output ONLY the translated Korean text."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
