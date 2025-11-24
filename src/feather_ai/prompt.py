"""
Easy way to create a multimodal prompt
"""

class Prompt:
    """
    The prompt class can be used to create a multimodal prompt with documents, images, etc.
    """
    def __init__(self, text: str):
        self.text = text