"""
Shared utilities for tab modules.

Provides helpers and utilities that tabs can use.
"""

from pathlib import Path


def format_help_html(markdown_text: str, height: str = "70vh") -> str:
    """
    Convert markdown text to formatted HTML for help display.
    
    Args:
        markdown_text: Markdown formatted text
        height: Container height (CSS value)
    
    Returns:
        HTML string
    """
    import markdown
    
    html = markdown.markdown(markdown_text, extensions=['extra', 'codehilite'])
    return f'''
    <div style="height:{height}; overflow-y: auto; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem;">
        {html}
    </div>
    '''
