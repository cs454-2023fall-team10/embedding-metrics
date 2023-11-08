def intent_prompt_from_user(intent):
    return {"messages": [{"role": "user", "content": intent}]}


def prompt_from_current_node(graph, node):
    edges = graph.edges_of(node)
    edge_labels = [edge.text() for edge in edges]

    messages = [
        {
            "role": "assistant",
            "content": node.text(),
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "exit",
                "description": "You are either satisfied or frustrated with the chatbot and want to exit",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summon",
                "description": "Summon a human agent to help you",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]

    if len(edges) > 0:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "move_to_node",
                    "description": "Navigate to another node of the chatbot based on its label",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {
                                "type": "string",
                                "description": "The label of the node to navigate to",
                                "enum": edge_labels,
                            }
                        },
                        "required": ["node"],
                    },
                },
            },
        )

    return {
        "messages": messages,
        "tools": tools,
    }


def run_conversation(graph, intent):
    import json
    import openai

    openai_client = openai.OpenAI()

    current_node = graph.root()
    messages = []

    print("Starting conversation...")

    # 1. System prompt
    system_prompt = [
        {
            "role": "system",
            "content": f"""
You are a chatbot assistant.
The user visited the website of a company "채널톡", which is a Korean IT startup.
Your goal is to navigate the user through the chatbot by choosing the right node to follow based on the user's intent.
You must prompt the user with a function call to move to the next node.
Try not to summon human agents if possible, but you can do so if you are stuck.
""",
        }
    ]
    messages.extend(system_prompt)

    # 2. Collect intent from user
    intent_prompt = intent_prompt_from_user(intent)
    messages.extend(intent_prompt["messages"])

    print(f"- User's intent: {intent_prompt['messages'][0]['content']}")

    path_length = 0

    while path_length < 10:
        path_length += 1

        # If there are no edges, then we have reached a leaf node
        # and the conversation is over
        edges = graph.edges_of(current_node)

        # 3. Prompt from current node
        prompt = prompt_from_current_node(graph, current_node)
        messages.extend(prompt["messages"])
        tools = prompt["tools"]

        print(f"- Chatbot's prompt: {prompt['messages'][0]['content']}")

        # 4. Generate response
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=tools,
        )

        # 5. Parse response
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        messages.append(response_message)

        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "move_to_node":
                print(f"- User selects {function_args['node']}")

                # 6. Navigate to next node
                edges = graph.edges_of(current_node)
                edge = None
                for _edge in edges:
                    if _edge.text() == function_args["node"]:
                        edge = _edge
                        break

                if edge is None:
                    print(f"ERROR: No edge with label {function_args['node']}")
                    break

                current_node = graph.node(edge.to_id)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": edge.text(),
                    },
                )

            elif function_name == "exit":
                print("- User exited the chatbot")
                break
            elif function_name == "summon":
                print("- User summoned a human agent")
                break
        else:
            print(f"ERROR: No tool call in response, response: {response}")
            break

    print("End of conversation.")
    print()


def usage():
    print(
        "Usage: python3 evalute.py <chatbot-filename> <intent-filename> <num-conversations>"
    )
    exit(1)


if __name__ == "__main__":
    import random
    import sys

    if len(sys.argv) != 4:
        usage()

    chatbot_filename = sys.argv[1]
    intent_filename = sys.argv[2]
    num_conversations = int(sys.argv[3])

    sys.path.append("chatbot-dataset")

    from chatbot import parse_from_file

    chatbot_graph = parse_from_file(chatbot_filename)

    with open(intent_filename, "r") as f:
        intents = f.readlines()
        random.shuffle(intents)
        for i in range(num_conversations):
            intent = intents[i].strip()
            run_conversation(chatbot_graph, intent)
