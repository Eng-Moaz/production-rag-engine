from dotenv import load_dotenv
import os
load_dotenv()

from langchain_groq import ChatGroq


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("Say 'setup complete'")
    print(response.content)

if __name__ == "__main__":
    main()
    