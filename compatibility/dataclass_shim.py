"""Enable dataclass usage in Python 3.6

Enables the usage of dataclasses in Python 3.6 without the need for installing the official backport
as a module.  Conditionally imports the official 3.6 backport of dataclasses depending on
interpreter version.

"""

import sys

if sys.version_info >= (3, 6) and sys.version_info < (3, 7):
    from compatibility.dataclasses_backport import *  # noqa: F401,F403
else:
    from dataclasses import *  # noqa: F401,F403
