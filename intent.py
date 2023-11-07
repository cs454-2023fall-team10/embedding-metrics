def generate_intent(graph):
    import openai

    openai_client = openai.OpenAI()

    name = graph.name

    messages = []

    # 1. System prompt
    system_prompt = [
        {
            "role": "system",
            "content": f"""
Assume that you are a user of a chatbot.
The chatbot is at the homepage of the Korean IT startup "채널톡".
Have conversation with the chatbot.
Think of a **very** specific situation related to the chatbot's purpose, which can be inferred from name {name}.
Be creative of a situation that is not too simple.
Assume korean users, so you should speak in Korean.
Express your intent in one sentence.
""",
        }
    ]
    messages.extend(system_prompt)

    # 2. Assistant prompt
    assistant_prompt = [
        {
            "role": "user",
            "content": f"Hello, I am a chatbot {name}. How can I help you?",
        }
    ]
    messages.extend(assistant_prompt)

    # 3. Get user input
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )
    print(response)

    # 4. Parse response
    response_message = response.choices[0].message
    return response_message.content


if __name__ == "__main__":
    import sys

    sys.path.append("chatbot-dataset")

    from chatbot import parse_from_file

    chatbot_graph = parse_from_file("chatbot-dataset/examples/jobs-homepage.json")

    intent = generate_intent(chatbot_graph)
    print(intent)
