import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

#variables de entorno
load_dotenv(override = True)

#Usamos deppseek
deepseek_provider = OpenAIProvider(
    base_url = 'https://api.deepseek.com',
    api_key = os.environ['DEEPSEEK_API_KEY']
)

deepseek_chat_model = OpenAIModel(
    'deepseek-chat',
    provider = deepseek_provider
)

#=========================================

#servidores MCP

exa_server = MCPServerStdio(
    'python',  #interprete
    ['busqueda.py']
)

python_tools_server = MCPServerStdio(
    'python',
    ['python_tool.py']
)

#==========================================
#Agente

agent = Agent(
    deepseek_chat_model,
    mcp_servers = [exa_server, python_tools_server],
    retries = 3
)

#=========================================
#Función principal

async def main():
    async with agent.run_mcp_servers():
        try:
            result = await agent.run("""
            Crea un gráfico de barras que muestre la población de las cinco ciudades más grandes del mundo
            """)
            print(result.output)
        except Exception as e:
            print("Error en la ejecución del agente:", e)
            result = "No se pudo generar la respuesta debido a un problema con la API."

if __name__ == '__main__':
    asyncio.run(main())