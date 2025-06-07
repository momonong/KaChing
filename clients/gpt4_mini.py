import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI 參數
endpoint = "https://refactor-rag.openai.azure.com/"
deployment = "gpt-4o-mini"
api_version = "2024-12-01-preview"
api_key = os.environ.get("AZURE_OPENAI_GPT4MINI_API_KEY")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

__all__ = ["client", "deployment"]
