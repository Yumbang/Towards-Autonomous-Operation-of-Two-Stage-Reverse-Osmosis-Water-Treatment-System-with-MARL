def save_as_markdown(content_dict: dict, file_path:str):
    """
    Save a given string to a file with a .md extension.

    Args:
    content (dict): The content to be saved.
    file_path (str): The path where the .md file will be saved, including the filename.

    Returns:
    None
    """
    # Ensure the file_path ends with .md
    if not file_path.endswith('.md'):
        file_path += '.md'
    
    # Open the file at the specified path in write mode
    with open(file_path, 'w', encoding='utf-8') as file:
        aggr_content = ""
        for element, content in content_dict.items():
            aggr_content += f"[{element}]\n{content}\n"
        file.write(aggr_content)