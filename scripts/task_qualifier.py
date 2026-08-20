import pandas as pd
import re
import json
import os

def qualify_tasks(input_file, output_dir):
    print(f"Reading data from {input_file}...")
    
    # Read the dataset (assuming it's an Excel or CSV file)
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
        
    qualified_tasks = []
    
    print(f"Total challenges found: {len(df)}")
    
    for index, row in df.iterrows():
        # 1. Filter only Development challenges
        if str(row.get('trackType', '')).strip().lower() != 'development':
            continue
            
        # 2. Check if it's a bug fix or First2Finish (more likely to have a clear bug to fix)
        if str(row.get('type', '')).strip().lower() not in ['first2finish', 'challenge']:
            continue
            
        description = str(row.get('description', ''))
        
        # 3. Extract GitHub Repository URL from description
        # Looking for patterns like https://github.com/...
        github_urls = re.findall(r'https://github\.com/[\w\-]+/[\w\-]+', description)
        if not github_urls:
            continue # No repo link found, skip
            
        # 4. Check if submission requires a patch (git patch file)
        requires_patch = 'patch' in description.lower() or 'git patch' in description.lower()
        
        # Extract a unique task ID
        task_id = f"tc_{row.get('id', index)}"
        
        qualified_task = {
            "task_id": task_id,
            "name": str(row.get('name', 'Unknown')),
            "trackType": row.get('trackType'),
            "type": row.get('type'),
            "technologies": str(row.get('technologies', '')),
            "github_repo": github_urls[0], # Take the first found repo
            "requires_patch": requires_patch,
            "description_excerpt": description[:200] + "..." if len(description) > 200 else description
        }
        
        qualified_tasks.append(qualified_task)
        
    print(f"Found {len(qualified_tasks)} potentially qualified tasks with GitHub links.")
    
    # Save the qualified list
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qualified_tasks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qualified_tasks, f, indent=4)
        
    print(f"Qualified tasks saved to: {output_path}")
    
    # Print first 5 as a preview
    print("\n--- Preview of first 5 qualified tasks ---")
    for task in qualified_tasks[:5]:
        print(f"ID: {task['task_id']} | Repo: {task['github_repo']} | Tech: {task['technologies']}")

if __name__ == "__main__":
    # Put the path to your downloaded Topcoder dataset here
    # It can be the xlsx file you showed the content of
    input_filepath = r"D:\raz\razieh\data\Challenges.xlsx" 
    
    # If it's a CSV, change the extension accordingly
    if not os.path.exists(input_filepath):
        input_filepath = r"D:\raz\razieh\data\Challenges.csv"
        
    output_directory = r"D:\raz\razieh\data\nsp_tasks"
    
    qualify_tasks(input_filepath, output_directory)