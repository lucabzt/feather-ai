"""
This class exposes some utility functions commonly used for AI agents
"""
import os
import logging
logger = logging.getLogger(__name__)
import inspect


def load_instruction_from_file(
        filename: str, default_instruction: str = "Default instruction."
) -> str:
    """Reads instruction text from a file relative to the caller's file location."""
    instruction = default_instruction
    try:
        # Get the caller's file path from the call stack
        caller_frame = inspect.stack()[1]
        caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))

        filepath = os.path.join(caller_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            instruction = f.read()
            logger.info("Successfully loaded instruction from %s", filename)
    except FileNotFoundError:
        logger.warning("Instruction file not found: %s. Using default.", filename)
    except Exception as e:
        logger.exception("ERROR loading instruction file %s: %s. Using default.", filename, e)
    return instruction