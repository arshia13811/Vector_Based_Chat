# Pinecone semaphore, number of matched to retrieve
# cutoff similarity score, and how much tokens as context
from pinecone import Pinecone, ServerlessSpec
import os
import tiktoken
from openai import OpenAI
import sys
from dotenv import load_dotenv
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np


index_name = 'semaphore'
context_cap_per_query = 30
match_min_score = 0.75    # similarity score  
context_tokens_per_query = 3000    

load_dotenv()

# OpenAI LLM model parameters
chat_engine_model = "gpt-3.5-turbo"
max_tokens_model = 4096
temperature = 0.2 
embed_model = "text-embedding-ada-002"
encoding_model_messages = "gpt-3.5-turbo-0301"
encoding_model_strings = "cl100k_base"
client = OpenAI(api_key=OPENAI_API_KEY)


# Connect with Pinecone db and index
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(index_name)

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_model_strings)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages. Compatible with  model """

    try:
        encoding = tiktoken.encoding_for_model(encoding_model_messages)
    except KeyError:
        encoding = tiktoken.get_encoding(encoding_model_strings)

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def get_prompt(query: str, context: str) -> str:
    """Return the prompt with query and context."""
    return (
        f"You are an expert at education and can give amazing consulting when it comes to courses to take, answer the question with taking the given context into account.\n" +
        f"Below you will find some context that may help. Ignore it if it seems irrelevant.\n\n" +
        f"Context:\n{context}" +
        f"\n\nTask: {query}\n\nResponse:"
    )

def get_message(role: str, content: str) -> dict:
    """Generate a message for OpenAI API completion."""
    return {"role": role, "content": content}


# embeddings=index_docs.encode_texts_in_batches 
# Load the tokenizer and model
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

def get_bert_embedding(query):
    """Generate BERT embeddings for a given query string."""
    inputs = tokenizer(query, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
    # Use the [CLS] token representation as the sentence embedding
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def get_context(query: str, max_tokens: int) -> str:
    """Generate context using BERT embeddings."""
    # Generate embedding for the query
    query_embedding = get_bert_embedding(query)

    # Assuming `index` is your Pinecone index already initialized
    search_response = index.query(vector=query_embedding[0].tolist(), top_k=context_cap_per_query, include_metadata=True)
    matches = search_response['matches']

    # filter and aggregate context
    usable_context = ""
    context_count = 0
    for i in range(0, len(matches)):
        if matches[i]['score'] < match_min_score:
            continue

        context = matches[i]['metadata']['text']
        token_count = num_tokens_from_string(usable_context + '\n---\n' + context)

        if token_count < context_tokens_per_query:
            usable_context += '\n---\n' + context 
            context_count += 1

    print(f"Found {context_count} contexts for your query")
    return usable_context


def complete(messages):
    """Query the OpenAI model. Returns the first answer. """

    res = client.chat.completions.create(
        model=chat_engine_model,
        messages=messages,
        temperature=temperature
    )
    return res.choices[0].message.content.strip()

# query = sys.argv[1]



def execution(query):
    while True:    
        context = get_context(query, context_tokens_per_query)
        prompt = get_prompt(query, context)

        # initialize messages list to send to OpenAI API
        messages = []
        messages.append(get_message('user', prompt))
        messages.append(get_message('system', 'You are a helpful assistant that helps learners choose the best course'))
    
        if num_tokens_from_messages(messages) >= max_tokens_model:
            raise Exception('Model token size limit reached') 

        answer = complete(messages)
        messages.append(get_message('assistant', answer))
        return answer
    
        # messages.pop() 
        # messages.append(get_message('assistant', answer))