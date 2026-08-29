from dotenv import load_dotenv
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI()

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)


from rag_helper import RAGBase


instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant_openai = RAGBase(
    index=index,
    llm_client=openai_client,
    instructions=instructions,
    model='gpt-5.4-mini',
    api_type='openai'
)

assistant_groq = RAGBase(
    index=index,
    llm_client=groq_client,
    instructions=instructions,
    model='qwen/qwen3.6-27b',
    api_type='groq'
)


answer = assistant_openai.rag('How do I run Ollama locally?')
print("OpenAI Answer:")
print(answer)

answer = assistant_groq.rag('How do I run Ollama locally?')
print("Groq Answer:")
print(answer)


messages = [
    {"role": "user", "content": "How do I run Olama locally?"}
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
)

print(response.output_text)


def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )


search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for answers to user questions",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}


response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

print(response.output)


import json

call = response.output[0]
args = json.loads(call.arguments)
print("Arguments:", args)

results = search(**args)
result_json = json.dumps(results, indent=2)
print("Search Results:", result_json)


messages.extend(response.output)

messages.append({
    "type" : "function_call_output",
    "call_id" : call.call_id,
    "output" : result_json
})


response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

print(response.output_text)


def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }

## Agentic loop starts from here.
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function.
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

question = "HOw to make coffee ?"

messages = [
    {"role": "developer", "content": instructions},
    {"role": "user", "content": question},
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool]
)


def agent_loop(instructions, question, model="gpt-5.4-mini") -> str:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]

    it = 1

    while True:
        print(f"iteration #{it}...")
        has_function_calls = False

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        messages.extend(response.output)

        for item in response.output:
            if item.type == "function_call":
                print("function_call:", item.name, item.arguments)
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True

            elif item.type == "message":
                print("ASSISTANT:")
                last_answer = item.content[0].text
                print(item.content[0].text)

        it = it + 1
        if has_function_calls == False:
            break

    return last_answer


agent_loop(instructions, question)


# ============================================================
# STUDY MAP — same code, explained
# ============================================================
#
# 1. search()
#    ---------------------------------------------------------
#    This is the REAL Python function that performs retrieval.
#
#        results = search(**args)
#
#    If:
#
#        args = {"query": "some question"}
#
#    then Python executes:
#
#        search(query="some question")
#
#    search() then calls:
#
#        index.search(...)
#
#
# 2. search_tool
#    ---------------------------------------------------------
#    This is NOT the search implementation.
#
#    It is the tool schema given to the LLM.
#
#    It tells the LLM:
#
#        "There is a function named search."
#
#    The LLM can therefore return:
#
#        function_call
#            name = search
#            arguments = {...}
#
#
# 3. make_call()
#    ---------------------------------------------------------
#    This is the bridge between the LLM and Python.
#
#        args = json.loads(call.arguments)
#
#    reads the arguments generated by the LLM.
#
#        result = search(**args)
#
#    executes the real Python search function.
#
#    Then it creates:
#
#        function_call_output
#
#    so the result can be returned to the LLM.
#
#
# 4. agent_loop()
#    ---------------------------------------------------------
#    This is the actual agentic controller.
#
#        while True:
#
#    means the process can repeat:
#
#        LLM
#          ↓
#        function_call
#          ↓
#        make_call()
#          ↓
#        search()
#          ↓
#        function_call_output
#          ↓
#        messages
#          ↓
#        LLM AGAIN
#
#    If the LLM requests another function:
#
#        has_function_calls = True
#
#    and the loop continues.
#
#    If the LLM returns a normal message and no function call:
#
#        has_function_calls = False
#
#    then:
#
#        break
#
#    and:
#
#        return last_answer
#
#
# ============================================================
# THE COMPLETE MENTAL MODEL
# ============================================================
#
#              USER QUESTION
#                    |
#                    v
#             +-------------+
#             |    LLM      |
#             | responses   |
#             |    .create  |
#             +------+------+
#                    |
#             function_call?
#                    |
#              +-----+-----+
#              |           |
#             YES          NO
#              |           |
#              v           v
#        +-----------+   FINAL
#        | make_call |
#        +-----+-----+
#              |
#              v
#        search(**args)
#              |
#              v
#        index.search()
#              |
#              v
#        search results
#              |
#              v
#      function_call_output
#              |
#              v
#          messages
#              |
#              v
#             LLM
#              |
#              +---------> repeat
#
#
# IMPORTANT:
#
# RAG      = retrieval of relevant context + using it to answer
# Search   = the retrieval operation in this example
# Tool     = a capability exposed to the LLM
# Function = the Python implementation of that capability
# Agent    = the LLM deciding what action/tool to use
# Loop     = repeatedly calling LLM -> tool -> result -> LLM
# History  = messages carrying the state between LLM calls
#
# Therefore:
#
#        LLM #1
#           ↓
#        search()
#           ↓
#        LLM #2
#
# is one tool-use cycle.
#
# With the while loop it can become:
#
#        LLM #1
#           ↓
#        search()
#           ↓
#        LLM #2
#           ↓
#        search()
#           ↓
#        LLM #3
#           ↓
#        FINAL ANSWER
#
# The number of LLM calls is therefore NOT fixed.
# It depends on how many tool calls the LLM requests before
# producing a final message.
