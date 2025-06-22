How to Contribute
=================

Thank you for considering contributing to **PhaseFieldX**! We welcome contributions in various forms, including reporting bugs, improving documentation, or writing code. Please read the guidelines below to ensure smooth collaboration.


Reporting Bugs
--------------

If you find a bug, please report it via the `Bug issue tracker <https://github.com/CastillonMiguel/phasefieldx/issues/new?labels=bug>`_.

Make sure to include:
- A clear description of the issue.
- Steps to reproduce the bug.
- Relevant environment details (e.g., operating system, version, etc.).


Contribution Workflow
---------------------

1. **Open an Issue First**:
   - Before making any changes (whether bug fixes or feature additions), please open an issue to discuss your idea.
   - Use the `Issue Tracker <https://github.com/CastillonMiguel/phasefieldx/issues>`_ to submit your ideas, bug reports, or requests for enhancements.


2. **Submit a Pull Request (PR)**:
   - Fork the repository and create a branch from ``main``.
   - Ensure your code adheres to our coding standards (see below).
   - Document your code thoroughly, including any necessary updates to the documentation.
   - Add tests for any new features or bug fixes.
   - When your work is ready, submit a PR and reference the relevant issue, if applicable.


Adding Tests
------------

To ensure the reliability of **PhaseFieldX**, we encourage you to add tests for any new features or bug fixes. Here’s how to do it:

- Create your test files in the appropriate folder, specifically in the ``test/`` folder. If you're unsure about where to place your tests, please open an issue related to it.
- Use **pytest** to write and run your tests.
- Each test function should start with ``test_`` to be recognized by pytest.
- Ensure that your tests cover the expected functionality and edge cases.
- Run your tests locally using the command:

  .. code-block:: bash

      pytest test/

- Verify that all tests pass before submitting your pull request.


Coding Style
------------

To maintain a consistent and clean codebase, PhaseFieldX follows the ``PEP8`` coding standard for Python code.  Please follow the `PEP8 Style Guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for detailed formatting instructions.


Docstrings
----------

PhaseFieldX uses the ``numpydoc`` style for its docstrings. Please follow the `numpydoc Style Guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for detailed formatting instructions.


Seeking Support
---------------

If you have any questions or need guidance, feel free to start a discussion on our `GitHub Discussions <https://github.com/CastillonMiguel/phasefieldx/discussions>`_ page.


Code of Conduct
---------------

We expect all contributors to follow our `Code of Conduct <https://github.com/CastillonMiguel/phasefieldx/blob/main/CODE_OF_CONDUCT.md>`_ to maintain a welcoming and inclusive environment.


Licensing
---------

All contributions to this project will be licensed under the MIT License, as specified in the repository. If you are contributing code that you did not write, you are responsible for ensuring that the original license is compatible with the MIT License. Alternatively, you must obtain permission from the original author to relicense the code under the MIT License before contributing.
