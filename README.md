![](resources/images/cover.png)

## Overview

The Digital Signage system is an intelligent advertising application that dynamically suggests advertisements based on demographic information. By using computer vision, decision trees, and data-driven predictions, this app enhances ad targeting for increased engagement and effectiveness.

GitHub Repository: [Digital Signage](https://github.com/Aamir0202/Intelligent-Digital-Signage)

## Features

-   Intelligent ad suggestions based on demography.
-   Configurable models (e.g., `imdb`).
-   Flexible input sources (e.g., video stream or file upload).
-   Two-level configuration system:
    1. `config.json` (user-defined overrides).
    2. `config.default.json` (default fallback).

## Requirements

-   Python 3.10+
-   Dependencies listed in `requirements.txt`

## Usage

### Running the Application

To run the app, use the following command:

```bash
python src/main.py [-h] [--config CONFIG] [--debug] [--model {imdb}] [--source {stream,upload}]
```

### Command-Line Options

-   `--config CONFIG`: Specify path to configuration file.
-   `--debug`: Enable debug mode.
-   `--model`: Select the model to use (`imdb`).

### Example

```bash
python src/main.py --model imdb --debug
```

### Custom Configuration (`config.json`)

To override defaults, specify the parameters you want to override in the `config.json` file. For example:

```json
{
    "model": "imdb"
}
```

## Supervisor

-   Dr. Shehzad Hasan

## Authors

-   Muhammad Aamir (CS-21029)
-   Mansoor Ahmed Memon (CS-21030)
-   Muhammad Kashif (CS-21032)

## License

This project is not licensed for public use, distribution, or modification. All rights are reserved by the authors.
