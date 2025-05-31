#!/usr/bin/env python3
"""
Beancount transaction processor
This script processes monthly transactions.
"""

import re
import os
import pathlib
import glob
import pandas as pd
import json
import requests
from typing import List
from datetime import datetime
import pytz
import pdfplumber
from beancount.core import data, amount
from beancount.core.number import Decimal
from beancount.parser.printer import format_entry
from beancount.loader import load_file
from beancount.core import account_types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuration variables
url = "http://192.168.5.44:11434/api/generate"
model = "llama3.1"
headers = {"Content-Type": "application/json"}
input_dir = "/home/errol/beancount/importer/00_inputs"
output_dir = "/home/errol/beancount/importer/01_outputs"
os.makedirs(output_dir, exist_ok=True)
start_date = "2025-03-22"
end_date = "2025-05-31"

# Define source file extraction functions
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from PDF, returning a list of text content per page"""
    page_contents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Processing PDF with {len(pdf.pages)} pages...")
        for page_num, page in enumerate(pdf.pages, 1):
            # print(f"Extracting text from page {page_num}...")
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                page_contents.append(text)
    
    return page_contents

def process_page(text: str, url: str, headers: dict) -> List[dict]:
    """Process a single page of text through Ollama with strict JSON validation"""
    prompt = f"""
    Extract all transactions from this bank statement page.
    Respond ONLY with a JSON array. Each object in the array must have these exact properties:
    [
        {{"Date": "YYYY-MM-DD", "Description": "string", "Amount": number}}
    ]
    
    --- Start of Page ---
    {text}
    --- End of Page ---
    """
    
    payload = {
        "model": model,
        "system": """You are a JSON-only API endpoint. 
        1. ONLY output valid JSON arrays
        2. Each object MUST have exactly: Date (YYYY-MM-DD), Description (string), Amount (number)
        3. NO explanation text
        4. NO markdown formatting
        5. NO natural language
        If you can't extract valid transactions, return an empty array []""",
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 2000,
        "stream": True  # Enable streaming for better response handling
    }
    
    try:
        full_response = ""
        response = requests.post(url, headers=headers, json=payload, stream=True)
        
        if response.status_code == 200:
            # Process the streamed response
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk.get("response", "")
            
            # Clean up the response to ensure valid JSON
            cleaned_response = full_response.strip()
            # Remove any non-JSON text before the first [
            cleaned_response = cleaned_response[cleaned_response.find("["):]
            # Remove any text after the last ]
            cleaned_response = cleaned_response[:cleaned_response.rfind("]")+1]
            
            try:
                # Validate JSON structure
                parsed_data = json.loads(cleaned_response)
                if not isinstance(parsed_data, list):
                    print("Warning: Response is not a JSON array")
                    return []
                    
                # Validate each transaction object
                valid_transactions = []
                for trans in parsed_data:
                    if all(key in trans for key in ["Date", "Description", "Amount"]):
                        valid_transactions.append(trans)
                    else:
                        print(f"Warning: Skipping invalid transaction: {trans}")
                
                return valid_transactions
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON response: {e}")
                print(f"Raw response: {cleaned_response[:100]}...")
                return []
                
    except Exception as e:
        print(f"Error processing page: {e}")
        return []
    

def import_csv_amazon(file_path):
    """Import Amazon Orders purchased with Gift Cards from csv file of Amazon Order history from "Amazon Order History Reporter" Chrome Extension into a standardized DataFrame."""
    if "Amazon" not in os.path.basename(file_path):
        print(f"Skipping file {file_path}: does not contain 'Amazon' in the file name.")
        return None
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(file_path)
        # Rename specific columns
        df = df.rename(columns={
            "order id": "OrderID",
            "order url": "OrderURL",
            "items": "Description",
            "to": "To",
            "date": "Date",
            "total": "Total",
            "shipping": "Shipping",
            "shipping_refund": "ShippingRefund",
            "gift": "Gift",
            "refund": "Refund",
            "payments": "Payments",
            "Invoice": "Invoice"
        })
        # Format the Date column to a datetime type
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors="coerce")

        # Convert numeric columns to numeric data types
        numeric_columns = ["Total", "Shipping", "ShippingRefund", "Gift", "Refund"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)  # Coerce errors to NaN and fill them with 0
    
        df['Amount'] = -df['Gift'] # Amount is the Gift amount only. We only want orders that were paid with Gift Cards
        df['TotalAmount'] = -(df['Total'] + df['Shipping'] - df['ShippingRefund'] + df['Gift']) # Total amount of the order

        # Select columsn into standardized DataFrame
        standardized_df = df[['Date', 'Description', 'Amount', 'TotalAmount', 'OrderURL', 'Refund']]
        standardized_df = standardized_df[standardized_df['Amount'] < 0] # Only select orders that were paid with Gift Cards
        
        # print(standardized_df.columns)
        # print(standardized_df)

        return standardized_df

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None to indicate failure

def import_json_woolworths(file_path, start_date=None, end_date=None):
    """
    Extracts transaction details, receipt totals, and amounts paid by gift cards from Woolworths JSON data.
    Optionally filters transactions based on a start and end date.

    For connector_version 3.2.0, where the download is an object with numbered keys.

    Args:
        file_path (str): Path to the JSON file.
        start_date (str): Start date in "YYYY-MM-DD" format (optional).
        end_date (str): End date in "YYYY-MM-DD" format (optional).

    Returns:
        pd.DataFrame: A DataFrame containing transaction details including receipt totals and gift card payments.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        
        connector_ver = data.get("connector_ver", None)
        if connector_ver != "3.2.0":
            print(f"Warning: Expected connector version 3.2.0, but found {connector_ver}. This function is designed for that version.")
            return None
        
        transactions = []
        perth_tz = pytz.timezone("Australia/Perth")  # Define Perth timezone

        # Handle the new structure where download is an object with numbered keys
        download_data = data.get("download", {})

        # print(f"Download data type: {type(download_data)}")
        # Convert the numbered keys to a list of entries
        entries = []
        if isinstance(download_data, dict):
            # Sort by key to maintain order
            for key in sorted(download_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                if isinstance(download_data[key], dict):
                    entries.append(download_data[key])
                    # print(f"Added entry for key {key}: {download_data[key]}")
        elif isinstance(download_data, list):
            # Fallback for old structure
            entries = download_data

        print(f"Download data type: {type(entries)}")
        print(f"Number of entries found: {len(entries)}")
        # print(entries)

        capture_time = data.get("captureTime", None)
        print(f"Capture Time: {capture_time}")

        # Parse captureTime to use as the reference datetime
        reference_datetime = None
        if capture_time:
            try:
                reference_datetime = datetime.fromisoformat(capture_time.replace("Z", "+00:00")).astimezone(perth_tz)
            except Exception as e:
                print(f"Error parsing captureTime: {e}")

        for entry in entries:
            # Skip entries that don't have the required structure
            if not isinstance(entry, dict):
                continue
                
            # Extract transaction-level details with safe default values
            transaction_id = entry.get("id", None)
            display_date = entry.get("displayDate", None)
            title = entry.get("title", None)
            description = entry.get("description", None)
            transaction = entry.get("transaction", {})
            transaction_type = entry.get("transactionType", None)

            # print(f"Processing entry with ID: {transaction_id}, Type: {transaction_type}, Title: {title}")
            
            # Skip non-purchase transactions
            if transaction_type != "purchase":
                continue
                
            # Extract amount from description if transaction is missing amountAsDollars
            amount = None
            if transaction and isinstance(transaction, dict):
                amount = transaction.get("amountAsDollars", "").replace("$", "") if transaction.get("amountAsDollars") else None

            # If no amount in transaction, try to extract from description
            if not amount and description:
                # Look for pattern like "$123.45" in description
                amount_match = re.search(r'\$(\d+\.\d+)', description)
                if amount_match:
                    amount = amount_match.group(1)
            
            origin = transaction.get("origin", None) if transaction else None
            # scraper = entry.get("scraper", {})
            # print(f"entry = {entry}")
            # print(f"scraper = {scraper}")
            # capture_time = entry.get("scraper",{}).get("captureTime",None)
 

            # Determine the transaction date using Display Date and Title
            transaction_date = None
            # print(f"Display Date: {display_date}, Title: {title}, Reference Datetime: {reference_datetime}")
            if display_date and reference_datetime:
                try:
                    # Parse day and month from display_date
                    day_month = datetime.strptime(display_date, "%a %d %b")  # Parse day and month
                    year = reference_datetime.year  # Default to the current year

                    if title == "Last Month":
                        # Handle transactions from the previous month
                        last_month = reference_datetime.month - 1 or 12
                        year -= 1 if last_month == 12 else 0
                        transaction_date = day_month.replace(year=year, month=last_month)
                    elif title == "This Month":
                        # Handle transactions from the current month
                        transaction_date = day_month.replace(year=year, month=reference_datetime.month)
                    else:
                        # Handle specific months and years in the title (e.g., "August 2023")
                        try:
                            # Extract the month and year from the title
                            month_year_match = re.match(r"(\w+)\s+(\d{4})", title)
                            if month_year_match:
                                month = datetime.strptime(month_year_match.group(1), "%B").month
                                year = int(month_year_match.group(2))
                                transaction_date = day_month.replace(year=year, month=month)
                            else:
                                # Handle specific months without a year in the title
                                month = datetime.strptime(title, "%B").month
                                transaction_date = day_month.replace(year=year, month=month)
                        except ValueError:
                            pass  # Ignore invalid formats
                    # print(f"Transaction date for entry {transaction_id}: {transaction_date}")
                except Exception as e:
                    print(f"Error parsing transaction date for entry {description}: {e}")

            # Extract eReceipt details safely
            ereceipt = entry.get("ereceipt", {}).get("activityDetails", {}).get("tabs", [])
            gift_card_payment = 0
            receipt_total = 0

            # print(f"ereceipt = {ereceipt}")

            if ereceipt:
                # print(f"eReceipt found for transaction {transaction_id}")
                receipt_details = list(ereceipt['0'].get("page", {}).get("details", {}).values())
                # print(f"Receipt details: {receipt_details}") 
                # Extract Receipt Total
                for detail in receipt_details:
                    if detail.get("__typename") == "ReceiptDetailsTotal":
                        receipt_total = float(detail.get("total", "0").replace("$", ""))
                        break
            # Extract Gift Card Payments
                for payment in receipt_details:
                    # print(f"Payment details: {payment}")
                    if payment.get("__typename") == "ReceiptDetailsPayments":
                        # Payments is also an object with numbered keys, convert to list
                        payment_methods = list(payment.get("payments", {}).values())
                        for method in payment_methods:
                            payment_description = method.get("description", "") or ""
                            alt_text = method.get("altText", "") or ""

                            # Check if "Gift" exists in either description or altText
                            if "Gift" in payment_description or "Gift" in alt_text:
                                gift_card_payment += float(method.get("amount", "0").replace("$", ""))
                
            # Only add transactions with valid amounts
            if amount:
                # print(f"Amount = {amount}") 
                # print(f"Amount type: {type(amount)}")
                # print(f"Amount as float: {float(amount)}")
                try:
                    amount_float = float(amount)
                    # print(amount_float)
                    transactions.append({
                        "TransactionID": transaction_id,
                        "TransactionType": transaction_type,
                        "Title": title,
                        "Date": transaction_date.strftime("%Y-%m-%d") if transaction_date else None,
                        "DisplayDate": display_date,
                        "Description": description,
                        "Origin": origin,
                        "TotalAmount": receipt_total if receipt_total > 0 else amount_float,
                        "Amount": gift_card_payment if gift_card_payment > 0 else 0,
                    })
                    # print(f"Added transaction {transaction_id} with amount {amount_float}")
                except ValueError:
                    print(f"Warning: Could not parse amount '{amount}' for transaction {transaction_id}")
                    continue
            else:
                print(f"Warning: No valid amount found for transaction {transaction_id}")

        # print(f"Transactions = {transactions}")
        # print(f"Number of transactions found: {len(transactions)}")
        # print(f"Transactions type: {type(transactions)}")

        # Convert the list of transactions to a DataFrame
        df = pd.DataFrame(transactions)

        # print(df)
        
        if df.empty:
            print("No valid transactions found in the file")
            return df

        # Format the Date column to a datetime type
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors="coerce")

        # Standardize the DataFrame
        standardized_df = df[['Date', 'Description', 'Amount', 'TotalAmount', 'Origin']].copy()
        # print(f"Standardized DataFrame:\n{standardized_df}") 
        # Filter rows where gift card payments were made (Amount > 0)
        standardized_df = standardized_df[standardized_df['Amount'] > 0].copy()

        # Convert Gift Card payments to negative amounts (expenses)
        if not standardized_df.empty:
            standardized_df['Amount'] = -standardized_df['Amount']

        # Apply the date filter on the standardized DataFrame
        if start_date:
            start_date_obj = pd.to_datetime(start_date)
            standardized_df = standardized_df[standardized_df['Date'] >= start_date_obj]
        if end_date:
            end_date_obj = pd.to_datetime(end_date)
            standardized_df = standardized_df[standardized_df['Date'] <= end_date_obj]

        return standardized_df
        # return entries

    except Exception as e:
        print(f"An error occurred: {e}")
        return None                

def standardize_transactions(transactions: list) -> list:
    """Standardize transaction format for both PDF and CSV sources"""
    standardized = []
    for trans in transactions:
        # Create standardized transaction dictionary
        std_trans = {
            "Date": trans.get("Date"),
            "Description": trans.get("Description"),
            "Amount": float(trans.get("Amount", 0)),
            "Source": trans.get("Source", "Unknown"),
            "URL": trans.get("OrderURL", ""),
            "Refund": trans.get("Refund", 0),
            "Category": trans.get("Category", "")  # Added for Woolworths categories
        }
        standardized.append(std_trans)
    return standardized

def process_file(file_path: str) -> None:
    """Process a single file (PDF, CSV, or JSON) and store results in a DataFrame and JSON"""
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    print("-" * 50)
    
    # Create matching JSON filename
    file_name = pathlib.Path(file_path).stem
    json_output_path = os.path.join(output_dir, f"{file_name}.json")
    
    # Process based on file type
    file_ext = pathlib.Path(file_path).suffix.lower()
    all_transactions = []
    
    if file_ext == '.pdf':
        pages = extract_text_from_pdf(file_path)
        print(f"Extracted {len(pages)} pages with content")
        
        for i, page_text in enumerate(pages, 1):
            print(f"Processing page {i}/{len(pages)}...")
            page_results = process_page(page_text, url, headers)
            if page_results:
                for transaction in page_results:
                    try:
                        date_str = transaction['Date']
                        # Try multiple date formats
                        date_obj = None
                        for fmt in ('%d-%b-%Y', '%d %b %Y', '%d %b %y','$d-%b-%y','%Y-%m-%d','%d-%m-%Y'):
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                # If year is 2-digit, convert to 4-digit (assume 2000+)
                                if fmt == '%d %b %y' and date_obj.year < 100:
                                    date_obj = date_obj.replace(year=2000 + date_obj.year)
                                break
                            except ValueError:
                                continue
                        if date_obj is None:
                            raise ValueError(f"Unrecognized date format: {date_str}")
                        transaction['Date'] = date_obj.strftime('%Y-%m-%d')
                        transaction['Source'] = 'Westpac'
                    except ValueError as e:
                        print(f"Warning: Invalid date format in transaction: {transaction['Date']}")
                    continue
            all_transactions.extend(page_results)
                
    elif file_ext == '.csv':
        df = import_csv_amazon(file_path)
        if df is not None:
            csv_transactions = df.to_dict('records')
            for trans in csv_transactions:
                trans['Date'] = trans['Date'].strftime('%Y-%m-%d')
                trans['Source'] = 'Amazon'
            all_transactions.extend(csv_transactions)
            
    elif file_ext == '.json':
        print(f"Processing Woolworths JSON file: {file_path}")
        df = import_json_woolworths(file_path, start_date, end_date)
        
        if df is not None and not df.empty:
            json_transactions = df.to_dict('records')
            for trans in json_transactions:
                trans['Source'] = 'Woolworths'
                if isinstance(trans['Date'], pd.Timestamp):
                    trans['Date'] = trans['Date'].strftime('%Y-%m-%d')
            all_transactions.extend(json_transactions)
            print(f"Processed {len(json_transactions)} Woolworths transactions")
        else:
            print(f"No valid transactions found in {file_path}")
    
    if all_transactions:
        # Standardize and sort transactions
        all_transactions = standardize_transactions(all_transactions)
        all_transactions.sort(key=lambda x: x['Date'])
        
        # Save to JSON file
        with open(json_output_path, 'w') as f:
            json.dump(all_transactions, f, indent=2)
        print(f"Saved transactions to: {json_output_path}")

def main():
    # Find all PDF, CSV, and JSON files in input directory
    input_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.csv")))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.json")))
    
    if not input_files:
        print(f"No PDF, CSV, or JSON files found in: {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for file_path in sorted(input_files):
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Print summary
    print("\nProcessing Complete")
    print("-" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files processed: {len(input_files)}")

# Process Transactions
def get_expense_accounts(bean_file: str) -> set:
    """Extract expense accounts from a Beancount file"""
    entries, errors, _ = load_file(bean_file)
    
    # Get all account open directives
    expense_accounts = set()
    for entry in entries:
        if isinstance(entry, data.Open) and entry.account.startswith('Expenses:'):
            expense_accounts.add(entry.account)
    
    return expense_accounts

def find_matching_transactions(beanfile_path, new_transaction_desc, threshold=0.5):
    """
    Finds transactions in the Beancount file where the 'src_desc' metadata field
    matches the description of a new transaction using TF-IDF and cosine similarity.

    Args:
        beanfile_path (str): Path to the Beancount file.
        new_transaction_desc (str): Description of the new transaction.
        threshold (float): Minimum cosine similarity score to consider a match (0-1).

    Returns:
        list: A list of tuples containing matching transactions and their similarity scores.
    """
    entries, errors, _ = load_file(beanfile_path)
    if errors:
        raise ValueError(f"Errors occurred while loading the Beancount file: {errors}")

    # Collect all transaction descriptions
    descriptions = []
    valid_entries = []
    
    for entry in entries:
        if isinstance(entry, data.Transaction):
            metadata = entry.meta or {}
            src_desc = str(metadata.get('src-desc', ''))
            if src_desc:
                descriptions.append(src_desc.lower())
                valid_entries.append(entry)

    if not descriptions:
        return []

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(lowercase=True, analyzer='char_wb', ngram_range=(2,3))
    tfidf_matrix = vectorizer.fit_transform(descriptions + [new_transaction_desc.lower()])
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[-1:], 
        tfidf_matrix[:-1]
    )[0]

    # Find matches above threshold
    matching_transactions = [
        (valid_entries[i], float(score))
        for i, score in enumerate(similarity_scores)
        if score >= threshold
    ]
    
    # Sort by similarity score in descending order
    matching_transactions.sort(key=lambda x: x[1], reverse=True)
    
    # print(f"\nFound {len(matching_transactions)} matching transactions for '{new_transaction_desc}'")
    return matching_transactions

def classify_expense(description: str, url: str, headers: dict, expense_accounts: set, matching_transactions: list) -> str:
    """
    Use Ollama to classify transaction description into a Beancount expense category
    """
    # Convert accounts set to formatted string for prompt
    account_list = '\n'.join(f'- {account}' for account in sorted(expense_accounts))
    
    matching_transactions_list = '\n'.join(f"\nMatch ratio: {score}%\n{format_entry(entry)}"
    for entry, score in matching_transactions)
    
    prompt = f"""
    Classify this transaction into a Beancount expense account category.
    Transaction: "{description}"
    
    Similar transactions from Beancount with similarity scores:
    {matching_transactions_list}
    
    Using the simmilar transactions as reference, use ONLY these categories (return ONLY the category name, no other text):
    {account_list}
    
    If unable to classify, return "Expenses:FIXME"
    """
    
    payload = {
        "model": model,
        "system": "You are a transaction classifier. Respond ONLY with the exact category name, no other text.",
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 100,
        "stream": True
    }
    
    try:
        full_response = ""
        response = requests.post(url, headers=headers, json=payload, stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk.get("response", "")
            
            # Clean up response and validate category
            category = full_response.strip()
            
            # Debug output
            # print(f"Raw response: '{full_response}'")
            # print(f"Cleaned category: '{category}'")
            # print(f"Is in expense_accounts: {category in expense_accounts}")
            # print(f"Available accounts: {expense_accounts}")
            
            return category if category in expense_accounts else 'Expenses:FIXME'
    except Exception as e:
        print(f"Error classifying transaction: {e}")
        return 'Expenses:FIXME'

def create_beancount_entries(
    json_file: str,
    bean_accounts_file: str,
    bean_transactions_file: str,
    url: str,
    headers: dict,
    output_file: str,
    account_mapping: dict
) -> None:
    """
    Create Beancount transactions from a JSON file and classify them using Ollama.
    Writes the output to a .bean file.

    Args:
        json_file (str): Path to the input JSON file.
        bean_accounts_file (str): Path to the Beancount file with expense account definitions.
        bean_transactions_file (str): Path to the Beancount file with existing transactions.
        url (str): URL of the Ollama model server.
        headers (dict): Headers for the Ollama HTTP request.
        output_file (str): Path to the output .bean file.
        assets_account (str): Default assets account for these transactions.
    """
    json_filename = os.path.basename(json_file)

    assets_account = next(
        (account for key, account in account_mapping.items() 
         if key.lower() in json_filename.lower()),
        'Assets:Cash'  # Default if no match found
    )

    entries = []
    expense_accounts = get_expense_accounts(bean_accounts_file)

    with open(json_file, 'r') as f:
        transactions = json.load(f)

    for txn in transactions:
        txn_date = datetime.strptime(txn['Date'], '%Y-%m-%d').date()
        description = txn['Description'].strip()
        txn_url = txn['URL']
        amt = Decimal(str(txn['Amount']))
        refund = Decimal(str(txn['Refund']))

        meta = data.new_metadata(json_file, 0)
        meta['src_desc'] = txn.get('Description', 'Unknown')
        meta['url'] = txn_url

        if amt < 0:
            matching_transactions = find_matching_transactions(bean_transactions_file, description)
            expense_account = classify_expense(description, url, headers, expense_accounts, matching_transactions)
            postings = [
                data.Posting(account=expense_account, units=amount.Amount(-amt, 'AUD'), cost=None, price=None, flag=None, meta=None),
                data.Posting(account=assets_account, units=amount.Amount(amt, 'AUD'), cost=None, price=None, flag=None, meta=None)
            ]
        elif refund > 0:
            matching_transactions = find_matching_transactions(bean_file, description)
            expense_account = classify_expense(description, url, headers, expense_accounts,matching_transactions)
            postings = [
                data.Posting(account=assets_account, units=amount.Amount(refund, 'AUD'), cost=None, price=None, flag=None, meta=None),
                data.Posting(account=expense_account, units=amount.Amount(-refund, 'AUD'), cost=None, price=None, flag=None, meta=None)
            ]
        else:
            continue

        txn_entry = data.Transaction(
            meta=meta,
            date=txn_date,
            flag='*',
            payee=txn.get('Source', 'Amazon'),
            narration=description,
            tags=set(),
            links=set(),
            postings=postings
        )

        entries.append(txn_entry)

    beancount_text = '\n\n'.join(format_entry(entry) for entry in entries)

    with open(output_file, 'w') as f:
        f.write(beancount_text)

    print(f"Beancount entries written to {output_file}")

ACCOUNT_MAPPING = {
  'Amazon-order-history': 'Assets:00-Personal:10-Non-Current-Assets:Gift-Cards:Amazon',
  'ING Blow-32970834': 'Assets:00-Personal:00-Current-Assets:ING:Blow-32970834',
  'Westpac Altitude-Credit-479923': 'Liabilities:01-Joint:Westpac:Altitude-Credit-479923',
  'Westpac Blow-524631': 'Assets:00-Personal:00-Current-Assets:Westpac:Blow-524631',
  'Westpac Joint-785669': 'Assets:01-Joint:00-Current-Assets:Westpac:Joint-785669',
  'Westpac Loan-343233': 'Liabilities:01-Joint:Westpac:Pyrus-Loan',
  'Woolworths-receipt': 'Assets:00-Personal:10-Non-Current-Assets:Gift-Cards:Woolworths'
}

if __name__ == "__main__":
    # Step 1: Extract transactions from source files
    main()
    
    # Step 2: Process transactions into Beancount format
    input_dir = '/home/errol/beancount/importer/01_outputs'  # Directory containing JSON files
    output_dir = '/home/errol/beancount/importer/02_beancount'
    bean_accounts_file= '/home/errol/beancount/ledger_prod/main.bean'
    bean_transactions_file = '/home/errol/beancount/ledger_prod/main.bean'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, '*.json'))

    # Process each JSON file and output a corresponding .bean file
    for json_file in json_files:
        base_name = os.path.basename(json_file)
        file_name_without_ext, _ = os.path.splitext(base_name)
        output_file = os.path.join(output_dir, f"{file_name_without_ext}.bean")
        
        create_beancount_entries(
            json_file=json_file,
            bean_accounts_file=bean_accounts_file,
            bean_transactions_file=bean_transactions_file,
            url=url,
            headers=headers,
            output_file=output_file,
            account_mapping=ACCOUNT_MAPPING
        )
        print(f"Processed {json_file} -> {output_file}")