from dotenv import load_dotenv
load_dotenv()

from langchain_openrouter import ChatOpenRouter


def main():
    llm = ChatOpenRouter(model="google/gemma-4-31b-it:free", temperature=0)
    response = llm.invoke("Say 'setup complete' in one word")
    print(response)

if __name__ == "__main__":
    main()
