"""
Module for logging conventions

```python
from zernikegrams.utils import log_config as logging
logger = logging.getLogger(__name__)
```
"""

import logging
from rich.logging import RichHandler
from rich.console import Console

format = "%(module)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format, handlers=[RichHandler(console=Console(width=120))])


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name)