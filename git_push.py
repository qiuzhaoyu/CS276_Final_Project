import os
import subprocess
from datetime import datetime

def auto_push(repo_path="./", branch="main"):  
    """
    Automate git add, commit, and push process.

    Parameters:
    repo_path (str): Path to the local git repository.
    branch (str): The branch to push changes to.
    """
    try:
        # Get current date and time for the commit message
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Automated commit on {current_datetime}"

        # Navigate to the repository path
        os.chdir(repo_path)
        print(f"Changed directory to: {repo_path}")

        # Add all changes to the staging area
        subprocess.run(["git", "add", "-A","."], check=True)
        print("Added all changes to the staging area.")

        # Commit changes with the generated message
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"Committed changes with message: '{commit_message}'")

        # Push changes to the specified remote branch
        subprocess.run(["git", "push"], check=True)
        print(f"Pushed changes to remote branch: {branch}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing git commands: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Define the path to the repository and the branch
    repo_path = "./"  # Change to your repository path
    branch = "main"   # Change to your target branch

    # Call the auto_push function with specified parameters
    auto_push(repo_path, branch)

if __name__ == "__main__":
    main()