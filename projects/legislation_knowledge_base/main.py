# Based on:
# https://medium.com/codex/gpt-4-chatbot-guide-mastering-embeddings-and-personalized-knowledge-bases-f58290e81cf4

# 1. Scrape source data from the web, divide it into sections and store it as a CSV file
# 2. Load the CSV file for further processing and set the correct indexes
# 3. Calculate vectors for each of the sections in the data file, using the embeddings endpoint
# 4. Search the relevant sections based on a prompt and the vectors (embeddings) we calculated
# 5. Answer the question in a chat session based on the context we provided

# Load unprocessed legislation data as a plain text file
from bs4 import BeautifulSoup
from dotenv import dotenv_values
import json
import numpy as np
import openai
import os
import pandas as pd
import requests
import tiktoken


config = {
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}


openai.api_key = os.getenv("OPENAI_API_KEY")

COMPLETIONS_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679&from=EN"

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def get_website(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def make_sections(content):
    # Now we split the legislation into sections, each section starts with the word "Article"
    search_document = content.split("HAVE ADOPTED THIS REGULATION:")
    articles = search_document[1].split("\nArticle")
    titled_articles = ["Article" + article for article in articles]

    return titled_articles[1:]


def get_section_titles(site, sections):
    section_titles_html = site.find_all(class_='sti-art')
    section_titles = [title.text for title in section_titles_html]

    return section_titles


def make_headings(size):
    return [f"Article {i}" for i in range(1, size + 1)]


def get_tokens(sections):
    encoder = tiktoken.encoding_for_model("gpt-4")

    return [len(encoder.encode(section)) for section in sections]


def make_data_frame(sections, section_titles, headings, tokens):
    return pd.DataFrame(list(zip(headings, section_titles, sections, tokens)),
                        columns=['Article', 'Title', 'Content', 'Tokens'])


def make_csv(csv_name):
    site = get_website(URL)
    legistlation = site.text

    sections = make_sections(legistlation)
    section_titles = get_section_titles(site, sections)
    headings = make_headings(len(sections))
    tokens = get_tokens(sections)

    data_frame = make_data_frame(
        sections, section_titles, headings, tokens)

    data_frame.to_csv(csv_name, index=False)


def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )

    return result["data"][0]["embedding"]


def compute_document_embeddings(data_frame: pd.DataFrame):
    return {
        row.Title: get_embedding(row.Content)
        for idx, row in data_frame.iterrows()
    }


def store_embeddings(embeddings: dict, filename: str):
    em_json = json.dumps(embeddings)

    with open(filename, 'w') as f:
        f.write(em_json)

    
def load_embeddings(filename: str) -> dict:
    try:

        with open(filename, 'r') as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        embeddings = {}

    return embeddings


def load_csv_to_data_frame(csv_name):
    data_frame = pd.read_csv(csv_name)
    data_frame.head()
    data_frame.set_index(['Title', 'Article'])

    return data_frame


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine
    similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_by_similarity(query: str, contexts: dict[tuple[str, str], np.array]) -> list[tuple[float, tuple[str, str]]]:
    """
    Find the query embedding for the supplied query, and compare it
    against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) 
        for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return [(similarity, index) for similarity, index in document_similarities]


def construct_propmt(question: str, context_embeddings: dict, df: pd.DataFrame):
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.Tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    return chosen_sections, chosen_sections_len


def answer_with_gpt(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[tuple[str, str], np.array],
    show_prompt: bool = False
) -> str:
    messages = [{
        "role" : "system",
        "content":"You are a GDPR chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"
    }]

    prompt, _ = construct_propmt(query, document_embeddings, df)

    if show_prompt:
        print(prompt)

    context = ""
    for article in prompt:
        context = context + article

    context = context + '\n\n --- \n\n' + query

    messages.append({
        "role": "user",
        "content": context
    })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    name = "legislation"
    csv_name = f'{name}.csv'

    data_frame = load_csv_to_data_frame(csv_name)
    if data_frame.empty:
        make_csv(csv_name)
        data_frame = load_csv_to_data_frame(csv_name)

    if data_frame.empty:
        print("No data frame found")
        exit()

    embeddings_file_name = f"{name}_embeddings.json"
    document_embeddings = load_embeddings(embeddings_file_name)

    if not document_embeddings:
        document_embeddings = compute_document_embeddings(data_frame)
        store_embeddings(document_embeddings, embeddings_file_name)


    # prompt the order_by_similarity function it will list all the articles
    # with similar vectors and sort them by relevancy
    result = order_by_similarity(
        "Can the commission implement acts for exchanging information?", 
        document_embeddings)

    [print(result) for result in result]