"""
This class exposes some utility functions commonly used for AI agents
"""
import os
import logging
logger = logging.getLogger(__name__)


def load_instruction_from_file(
    filename: str, default_instruction: str = "Default instruction."
) -> str:
    """Reads instruction text from a single file relative to the caller's working directory."""
    instruction = default_instruction
    try:
        # Use current working directory instead of script location
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, "r", encoding="utf-8") as f:
            instruction = f.read()
            logger.info("Successfully loaded instruction from %s", filename)
    except FileNotFoundError:
        logger.warning("Instruction file not found: %s. Using default.", filepath)
    except Exception as e:
        logger.exception("ERROR loading instruction file %s: %s. Using default.", filepath, e)
    return instruction