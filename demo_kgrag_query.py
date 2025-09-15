import asyncio
from demo_kgrag_ollama import kgrag_ollama


async def query(prompt: str):
    response = await kgrag_ollama.query(prompt)
    return response


async def main():
    result = await query(
        "Come misurare empiricamente il “consumo effettivo” "
        "e stimare la CAWF?"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
