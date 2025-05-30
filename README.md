# Beancount Processor

A Python script for processing and importing financial data into Beancount format.

## Overview

This tool helps automate the process of converting financial transaction data into Beancount's double-entry accounting format.

## Features

- Processes financial transaction files
- Converts data to Beancount format
- Handles multiple file types and formats
- Provides data validation and error checking

## Usage

Setup virtual environment
```bash
python3 -m venv beancount-venv
source beancount-venv/bin/activate
```

Install Python Pre-requisites
```bash
pip install --upgrade pip
pip install -r /home/errol/beancount/importer/10_scripts/requirements.txt
```

Run script
```bash
python beancount_processor.py [options] input_file
```

## Requirements

- Python 3.x
- Beancount library
- Additional dependencies as specified in requirements.txt

## Installation

1. Clone or download the script
2. Install required dependencies
3. Configure your account mappings
4. Run the processor on your transaction files

## Configuration

Edit the configuration section in the script to match your:
- Account names
- Currency settings
- File format preferences

## Output

The script generates properly formatted Beancount entries that can be included in your main accounting file.