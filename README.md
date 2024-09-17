# summarize-hnews

Summarize Hacker News by first extracting/summarizing (w/OpenAI) each thread and all resources before creating overall outline

# Installation

1. python -m venv venv
2. (Win11 PS CLI) venv/Scripts/Activate.ps1 
3. pip install -r requirements
4. python -m spacy download en_core_web_sm

# Execution

1. Open 'scrape_hnews_ver1.py'
2. Set URL_TARGET to the hnews discussion URL
3. Set URL_SUMMARY to the output file
4. python scrape_hnews_ver1.py
5. Enter OpenAI API Key at CLI
6. Results in URL_SUMMARY file
