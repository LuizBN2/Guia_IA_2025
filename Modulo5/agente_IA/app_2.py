import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Proveedor: OpenRouter
openrouter_provider = OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ['OPENROUTER_API_KEY']
)

# Modelo: GPT-3.5-turbo (u otro disponible en OpenRouter)
openrouter_model = OpenAIModel(
    'openai/gpt-3.5-turbo',
    provider=openrouter_provider
)

# Servidores MCP
exa_server = MCPServerStdio(
    'python',
    ['busqueda.py']
)

python_tools_server = MCPServerStdio(
    'python',
    ['python_tool.py']
)

# Crear agente con servidores MCP
agent = Agent(
    openrouter_model,
    mcp_servers=[exa_server, python_tools_server],
    retries=3
)

# Función principal
async def main():
    async with agent.run_mcp_servers():
        try:
            result = await agent.run("""
            Crea un gráfico de barras que muestre la población de las cinco ciudades más grandes del mundo.
            """)
            print(result.output)
        except Exception as e:
            print("Error en la ejecución del agente:", e)

if __name__ == '__main__':
    asyncio.run(main())
