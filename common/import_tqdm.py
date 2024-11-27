import sys as __sys
if 'ipykernel' in __sys.modules:
    # If using tqdm.notebook in Jupyter Notebook
    from tqdm.notebook import tqdm
    # from tqdm.asyncio import tqdm
else:
    # If using tqdm in a regular Python script
    from tqdm import tqdm