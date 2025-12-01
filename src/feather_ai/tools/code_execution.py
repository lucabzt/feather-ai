"""
Defines tools for native code execution.
"""

from langchain_experimental.utilities import PythonREPL

_python_repl = None

def code_execution_python(python_code: str) -> str:
    """
    A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.
    Returns:
        Python output as a string
    """
    global _python_repl
    if not _python_repl:
        _python_repl = PythonREPL()

    return _python_repl.run(python_code)