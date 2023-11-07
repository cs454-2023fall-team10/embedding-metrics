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
You are a user visiting the webpage of the Korean IT startup "채널톡".
Have conversation with the chatbot.
Think of a **very** specific situation related to the chatbot's name, {name}.
Be **creative**, but keep the topic related to {name}.
Express your intent in Korean, in **one** sentence.
(You don't need to greet or say that you are visiting the homepage. Be brief.)
""",
        }
    ]
    messages.extend(system_prompt)

    # 2. Assistant prompt
    assistant_prompt = [
        {
            "role": "user",
            "content": graph.root().text(),
        }
    ]
    messages.extend(assistant_prompt)

    # 3. Get user input
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )

    # 4. Parse response
    response_message = response.choices[0].message
    return response_message.content


def usage():
    print("Usage: python intent.py <graph_filename> <out_filename> <num_samples>")
    exit(1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        usage()

    graph_filename = sys.argv[1]
    out_filename = sys.argv[2]
    num_samples = int(sys.argv[3])

    sys.path.append("chatbot-dataset")

    from chatbot import parse_from_file

    chatbot_graph = parse_from_file(graph_filename)

    # append
    with open(out_filename, "a") as f:
        for i in range(num_samples):
            print(f"Generating intent {i + 1}/{num_samples}")
            intent = generate_intent(chatbot_graph)
            f.write(intent + "\n")
            f.flush()

    print(f"Generated {num_samples} intents and saved to {out_filename}")
