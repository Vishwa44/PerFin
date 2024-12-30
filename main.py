import PyPDF2
import anthropic
import os
import csv
from datetime import datetime
import pandas as pd

def analyze_bank_statement(pdf_path, output_csv_path):
    """
    Analyze a bank statement PDF and export categorized transactions to CSV.
    
    Args:
        pdf_path (str): Path to the PDF bank statement
        output_csv_path (str): Path where the CSV will be saved
    """
    # Read the PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Extract text from all pages except the first
        text_content = ""
        for page_num in range(1, len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text() + "\n\n"
    
    # Initialize Claude client
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create the prompt
    categorization_prompt = """Can you read this bank statement and classify each transaction in categories mentioned below, give it in a CSV format. Below are categories and the kind of transactions to be classified in those categories. Avoid summing the transaction amount. Avoid clubbing transactions in a single row. Look at transactions only in the purchases sections. Account for all the transactions in the statement. Sort the transactions chronologically.

Categories:
- Travel: Flights, Hotels, Any transactions in non-US currencies
- Local Travel (Car + Cabs): Uber, Lyft, Gas, Automobile Service, Car Cleaning
- Shopping Home: Grocery Chains, Indian stores, Amazon transactions
- Personal Shopping: Hair cuts, smoke shop, retail brands
- Personal Services: Trade Subscriptions
- Subscriptions: Other subscriptions
- Subscriptions Home: Trash
- Eat + Drink Out: Restaurants, Bars, Doordash, Uber Eats
- Medical
- Work-outs/Fitness
- Misc: Any refunds, uncategorizable transactions

Here's the bank statement:

{text_content}

Please provide the output in CSV format with the following columns:
Date,Description,Amount,Category"""
    
    # Make API call to Claude
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=3000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": categorization_prompt
            }
        ]
    )
    
    # Extract CSV content from Claude's response
    claude_response = message.content
    
    # Find the CSV portion of the response
    # Usually it starts after the header line
    csv_start = claude_response.find("Date,Description,Amount,Category")
    if csv_start == -1:
        raise ValueError("Could not find CSV data in Claude's response")
    
    csv_content = claude_response[csv_start:]
    
    # Convert the CSV string to a pandas DataFrame
    # Using StringIO to read the CSV string
    from io import StringIO
    df = pd.read_csv(StringIO(csv_content))
    
    # Sort by date if needed
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Export to CSV
    df.to_csv(output_csv_path, index=False)
    
    return df

def validate_results(df):
    """
    Validate the analysis results.
    
    Args:
        df (pandas.DataFrame): The analyzed transactions
    
    Returns:
        dict: Validation results
    """
    validation = {
        'total_transactions': len(df),
        'categories_found': df['Category'].unique().tolist(),
        'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
        'total_amount': df['Amount'].sum()
    }
    return validation

# Example usage
if __name__ == "__main__":
    # Set up your API key
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    # Define paths
    pdf_path = "bank_statement.pdf"
    output_csv_path = "categorized_transactions.csv"
    
    try:
        # Analyze the bank statement and export to CSV
        df = analyze_bank_statement(pdf_path, output_csv_path)
        
        # Validate results
        validation = validate_results(df)
        
        # Print summary
        print("\nAnalysis Complete!")
        print("-" * 50)
        print(f"Total transactions processed: {validation['total_transactions']}")
        print(f"Date range: {validation['date_range']}")
        print(f"Categories found: {', '.join(validation['categories_found'])}")
        print(f"\nResults have been exported to: {output_csv_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")