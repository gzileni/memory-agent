import asyncio
from demo_kgrag_ollama import kgrag_ollama


async def main():
    prompt = (
        "How Big Data Dilutes Cognitive Resources, Interferes with Rational "
        "Decision-making and Affects Wealth Distribution ?"
    )
    response = await kgrag_ollama.query(prompt)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
