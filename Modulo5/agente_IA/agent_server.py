import subprocess
import atexit

# 🚀 1. Lanzar `busqueda.py` como proceso MCP en segundo plano
busqueda_process = subprocess.Popen(["python", "busqueda.py"])
tools_process = subprocess.Popen(["python", "python_tool.py"])

# 🧹 2. Asegurar que se cierre al finalizar
atexit.register(lambda: busqueda_process.terminate())

# 🧠 3. Configurar y lanzar el agente
from pydantic_ai import Agent
from pydantic_ai import tool

@tool(mcp_server="busqueda.py")  # nombre del script del servidor
def search_web(query: str, num_results: int = 5) -> str:
    """Busca información en la web usando Exa."""
    return "Llamando al servidor MCP..." #

agent = Agent(tools=[search_web])
agent.serve()
