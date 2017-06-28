# typhon - Tools for atmospheric research

## Requirements
Typhon requires Python version 3.5 or higher. The recommended way to get Python
is through [Anaconda]. But of course, any other Python distribution is also
working.

## Installation
The easiest way to develop typhon is to install the local working copy in your
Python environment. This can be done using ``pip``:
```bash
pip install --user --editable .
```

This will install the package in editable mode (develop mode) in the user's
home directory. That way, local changes to the package are directly available
in the current environment.

## Testing
Typhon contains a simple testing framework using [Nosetests]. It is good
practice to write code for all your functions and classes. Those tests may not
be too extensive but should cover the basic use cases to ensure correct
behavior through further development of the package.

Tests can be run on the command line...
```bash
nosetests typhon
```
or using the Python interpreter:
```python
import typhon
typhon.test()
```

## Documentation
The documentation of the project is created with [Sphinx]. You can use the
enclose Makefile to build your own documentation:
```bash
cd doc
make html
```

The latest version [Typhon Docs] is also accessible online.

[Sphinx]: www.sphinx-doc.org
[Anaconda]: https://www.continuum.io/downloads
[Typhon Docs]: https://radiativetransfer.org/misc/typhon/doc-trunk
[Nosetests]: http://nose.readthedocs.io/