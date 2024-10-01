import os
import re
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
import logging
from typing import List, Dict, Tuple
import asyncio
import aiohttp
import sys
import spacy
import getpass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

URL_TARGET = os.getenv('URL_TARGET', 'https://news.ycombinator.com/item?id=40198766')
URL_TARGET = os.getenv('URL_TARGET', 'https://news.ycombinator.com/item?id=40192204')
URL_TARGET = os.getenv('URL_TARGET', 'https://news.ycombinator.com/item?id=40291577')
URL_TARGET = os.getenv('URL_TARGET', 'https://news.ycombinator.com/item?id=40515465')
URL_TARGET = os.getenv('URL_TARGET', 'https://news.ycombinator.com/item?id=41628167')

URL_SUMMARY = os.getenv('URL_SUMMARY', 'ycombinatornews_40198766.txt')
URL_SUMMARY = os.getenv('URL_SUMMARY', 'ycombinatornews_40192204.txt')
URL_SUMMARY = os.getenv('URL_SUMMARY', 'ycombinatornews_40291577.txt')
URL_SUMMARY = os.getenv('URL_SUMMARY', 'ycombinatornews_40515465.txt')
URL_SUMMARY = os.getenv('URL_SUMMARY', 'ycombinatornews_41628167.txt')


OPENAI_API_KEY = getpass.getpass("Enter OpenAI API Key: ") # os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

async def fetch_web_page(url: str, session: aiohttp.ClientSession) -> str:
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        logging.error(f"Error fetching {url}: {e}")
        raise

def parse_threads(html_content: str) -> List[Dict]:
    soup = BeautifulSoup(html_content, 'html.parser')
    threads = []

    try:
        comments = soup.find_all('tr', class_='athing comtr')
        for comment in comments:
            comment_id = comment.get('id')
            indent = int(comment.find('td', class_='ind').find('img').get('width', 0)) // 40
            content = comment.find('div', class_='comment')
            
            if content:
                threads.append({
                    'id': comment_id,
                    'indent': indent,
                    'content': content.get_text(strip=True),
                    'replies': []
                })
    except AttributeError as e:
        logging.error(f"Error parsing HTML: {e}")
        raise

    return threads

def build_thread_tree(threads: List[Dict]) -> List[Dict]:
    thread_tree = []
    stack = []

    for thread in threads:
        while stack and stack[-1]['indent'] >= thread['indent']:
            stack.pop()
        
        if stack:
            stack[-1]['replies'].append(thread)
        else:
            thread_tree.append(thread)
        
        stack.append(thread)

    return thread_tree

def extract_resources(text: str) -> List[Tuple[str, str, str]]:
    doc = nlp(text)
    resources = []
    
    # Extract URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = re.findall(url_pattern, text)
    for url in urls:
        resources.append(("URL", url, url))
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "PRODUCT", "WORK_OF_ART"]:
            resources.append((ent.label_, ent.text, ""))
    
    return resources

async def summarize_thread(thread: Dict, session: aiohttp.ClientSession) -> Tuple[str, List[Tuple[str, str, str]]]:
    prompt = f"Summarize the following discussion thread concisely:\n\n{thread['content']}"
    
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            }
        ) as response:
            response.raise_for_status()
            result = await response.json()
            summary = result['choices'][0]['message']['content'].strip()
            resources = extract_resources(thread['content'])
            return summary, resources
    except aiohttp.ClientError as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise

async def summarize_threads(threads: List[Dict]) -> Tuple[List[Dict], Dict[int, Tuple[str, str, str]]]:
    async with aiohttp.ClientSession() as session:
        tasks = [summarize_thread(thread, session) for thread in threads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_resources = {}
    resource_counter = 1

    for thread, result in zip(threads, results):
        if isinstance(result, Exception):
            logging.error(f"Error summarizing thread {thread['id']}: {result}")
            thread['summary'] = "Error in summarization"
            thread['resources'] = []
        else:
            summary, resources = result
            thread['summary'] = summary
            thread['resources'] = resources
            for resource in resources:
                all_resources[resource_counter] = resource
                resource_counter += 1
    
    return threads, all_resources

def combine_summaries(threads: List[Dict]) -> str:
    combined_summary = "Individual Thread Summaries:\n\n"
    
    for thread in threads:
        combined_summary += f"Thread ID: {thread['id']}\nSummary: {thread['summary']}\n\n"
    
    return combined_summary

async def create_outline_synthesis(threads: List[Dict], session: aiohttp.ClientSession) -> str:
    all_summaries = "\n".join([thread['summary'] for thread in threads])
    prompt = f"""Create an outline synthesizing all the topics from the following summaries. 
    Include supporting details like:
    a. Concise description
    b. Pros
    c. Cons
    d. Related resources (if any)

    Summaries:
    {all_summaries}
    """

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['message']['content'].strip()
    except aiohttp.ClientError as e:
        logging.error(f"Error calling OpenAI API for synthesis: {e}")
        raise

def format_resources(resources: Dict[int, Tuple[str, str, str]]) -> str:
    formatted = "Collected Resources:\n\n"
    for key, (res_type, res_name, res_url) in resources.items():
        formatted += f"{key}. Type: {res_type}, Name: {res_name}"
        if res_url:
            formatted += f", URL: {res_url}"
        formatted += "\n"
    return formatted

def save_summary_to_file(thread_summaries: str, outline_synthesis: str, resources: str, file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("--- Thread Summaries ---\n\n")
            file.write(thread_summaries)
            file.write("\n--- Overall Outline Synthesis ---\n\n")
            file.write(outline_synthesis)
            file.write("\n--- Collected Resources ---\n\n")
            file.write(resources)
        logging.info(f"Summary saved to {file_path}")
    except IOError as e:
        logging.error(f"Error saving summary to file: {e}")
        raise

async def main():
    try:
        async with aiohttp.ClientSession() as session:
            html_content = await fetch_web_page(URL_TARGET, session)
        
        threads = parse_threads(html_content)
        thread_tree = build_thread_tree(threads)
        summarized_threads, all_resources = await summarize_threads(thread_tree)
        
        thread_summaries = combine_summaries(summarized_threads)
        
        async with aiohttp.ClientSession() as session:
            outline_synthesis = await create_outline_synthesis(summarized_threads, session)
        
        formatted_resources = format_resources(all_resources)
        
        save_summary_to_file(thread_summaries, outline_synthesis, formatted_resources, URL_SUMMARY)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())