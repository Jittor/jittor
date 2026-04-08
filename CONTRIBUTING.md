# Contributing to Jittor

Thank you for your interest in contributing to jittor! We welcome contributions from the community to help make jittor better.

Whether you are fixing a bug, adding a new operator, or improving documentation, this guide will help you get started.

## 0. Where to Start?

If you are new to Jittor, check our issue tracker.

*   [Find an Issue to Contribute](https://github.com/Jittor/jittor/issues)

[MAINTAINER TODO: Label some issues as 'good first issue', or use tools like [gfi-bot](https://github.com/osslab-pku/gfi-bot) to label automatically.]

If you have a new idea or feature request, please open an issue first to discuss it with the maintainers before writing code.

## 1. Development Environment Setup

Unlike installing Jittor as a user, developers need to set up an environment that supports compiling from source and debugging.

### Prerequisites

[MAINTAINER TODO: Insert specific prerequisites, e.g., python version >= 3.7.]

### Installation from Source

[MAINTAINER TODO: Insert specific command to install, e.g., python3 -m pip install -e .] 

Please see [README#manual-install](README.md#manual-install).


## 2. Contribution Workflow

We follow the standard GitHub Pull Request workflow. Please following these steps:

1.  **Create a Branch**: Always work on a new branch, not `master`.
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Code & Commit**: Keep your changes focused and your commit messages clear.
3.  **Push**: Push the branch to your forked repository.
4.  **Open a PR**: [Submit a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to the main Jittor repository. Ensure that your Pull Request has a clear title, a detailed description of what the PR is doing, and any additional information such as linked issue.


## 3. Pre-Commit Checks

Before you submit your PR, please ensure you have completed the following checks to ensure code quality.

### Test Your Changes
Please run the tests to ensure your changes don't break existing functionality.

[MAINTAINER TODO: Insert command to run core tests, e.g., python3 -m jittor.test.test_core]

### Code Style
*   [MAINTAINER TODO: If you have a specific linter command, list it here, e.g., Python use [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards]


## 4. Getting Help

If you run into issues during the setup or contribution process:
*   Check the [Issues](https://github.com/Jittor/jittor/issues) page.
*   Contact us via [MAINTAINER TODO: Insert Email or Link to QQ Group/Forum].

---
*Note: This document is a starting point for governance. We encourage the community to help refine it.*