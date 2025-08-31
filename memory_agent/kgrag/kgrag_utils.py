import markdown
from bs4 import BeautifulSoup
from bs4.element import Tag


def print_progress_bar(
    iteration,
    total,
    prefix='',
    length=80
):
    """
    Prints a progress bar to the console.
    Args:
        iteration (int): Current iteration number.
        total (int): Total number of iterations.
        prefix (str): Prefix string to display before the progress bar.
        length (int): Length of the progress bar.
    """
    percent = 100 * iteration // total
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # end='\r' mantiene tutto sulla stessa riga, flush=True aggiorna subito
    print(f'\r{prefix}: {percent:3d}%|{bar}| {iteration}/{total}',
          end='',
          flush=True)
    if iteration == total:
        print()


def markdown_to_html_no_headers(md_text):
    """
    Converts markdown text to HTML and removes header tags.
    """
    # Convert markdown to HTML
    html = markdown.markdown(md_text)
    # Parse HTML and remove header tags
    soup = BeautifulSoup(html, "html.parser")
    for header_tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        # Replace header with its contents (remove the tag, keep the text)
        if isinstance(header_tag, Tag):
            header_tag.unwrap()
    return str(soup)
