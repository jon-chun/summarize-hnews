import os
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
import logging
from typing import List, Dict
import asyncio
import aiohttp
import sys
import getpass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

URL_TARGET = os.getenv('URL_TARGET', 'https://news.ycombinator.com/item?id=40198766')
URL_SUMMARY = os.getenv('URL_SUMMARY', 'ycombinatornews_40198766.txt')
OPENAI_API_KEY = getpass.getpass("Enter OpenAI API Key: ") # os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

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

async def summarize_thread(thread: Dict, session: aiohttp.ClientSession) -> str:
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
            return result['choices'][0]['message']['content'].strip()
    except aiohttp.ClientError as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise

async def summarize_threads(threads: List[Dict]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [summarize_thread(thread, session) for thread in threads]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
    
    for thread, summary in zip(threads, summaries):
        if isinstance(summary, Exception):
            logging.error(f"Error summarizing thread {thread['id']}: {summary}")
            thread['summary'] = "Error in summarization"
        else:
            thread['summary'] = summary
    
    return threads

def combine_summaries(threads: List[Dict]) -> str:
    combined_summary = "Combined Summary of Threads:\n\n"
    common_topics = {}
    
    for thread in threads:
        combined_summary += f"Thread ID: {thread['id']}\nSummary: {thread['summary']}\n\n"
        
        words = thread['summary'].lower().split()
        for word in words:
            if len(word) > 5:
                common_topics[word] = common_topics.get(word, 0) + 1
    
    combined_summary += "Common Topics:\n"
    for topic, count in sorted(common_topics.items(), key=lambda x: x[1], reverse=True)[:5]:
        combined_summary += f"- {topic}: mentioned {count} times\n"
    
    return combined_summary

def save_summary_to_file(summary: str, file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(summary)
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
        summarized_threads = await summarize_threads(thread_tree)
        combined_summary = combine_summaries(summarized_threads)
        save_summary_to_file(combined_summary, URL_SUMMARY)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())