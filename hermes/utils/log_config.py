"""
Module for logging conventions

```python
from src.utils import log_config as logging
logger = logging.getLogger(__name__)
```
"""

import logging
from rich.logging import RichHandler


format = "%(module)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format, handlers=[RichHandler()])


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name)