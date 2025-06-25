from mcp.server.fastmcp import FastMCP
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback
import os
from datetime import datetime

mcp = FastMCP('python_tools')

class PythonREPL:    #REPL -> read, eval, print, loop
    def run(self, code):
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()  #guardamos la salida en memoria sin mostrar
        try:
            exec(code, globals())   #use el espacio de memoria global para las variables
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f'Error: {str(e)}\n{traceback.format_exc()}'

repl = PythonREPL()   #Objeto de la clase


#==========================================================

@mcp.tool()
async def python_repl(code: str) -> str:
    """Execute Python code."""
    return repl.run(code)

@mcp.tool()
async def data_visualization(code: str) -> str:
    """Execute Python code. Use matplotlib for visualization."""
    try:
        repl.run(code)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()   #close defigure to free memory
        return f'data:image/png;base64, {img_str}'
    except Exception as e:
        return f'error creating chart: {str(e)}'

if __name__ == '__main__':
    mcp.run()
