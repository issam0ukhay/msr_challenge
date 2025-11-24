import os
import subprocess
from pathlib import Path
import shutil
import pandas as pd
from huggingface_hub import login
login("")

import pandas as pd
all_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")

# ---------- CONFIG ----------
BASE_DIR = Path("./repos")  # where repos will be cloned
RESULTS_CSV = Path("pr_commits_results.csv")
# -----------------------------


def run_cmd(cmd, cwd=None):
    """Run a shell command and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def repo_local_path(repo_url: str) -> Path:
    """
    Derive a local folder name from the repo URL.
    Example: https://github.com/django/django.git -> repos/django-django
    """
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid repo URL: {repo_url}")
    owner = parts[-2]
    name = parts[-1]
    return BASE_DIR / f"{owner}-{name}"


def clone_or_update_repo(repo_url: str, path: Path) -> bool:
    """Clone the repo if not exists; otherwise do a git fetch.
       Returns True on success, False on failure.
    """
    if not path.exists():
        print(f"[INFO] Cloning {repo_url} -> {path}")
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        out, err, code = run_cmd(["git", "clone", repo_url, str(path)])
        if code != 0:
            print(f"[ERROR] Error cloning {repo_url}:\n{err}")
            return False
        return True
    else:
        print(f"[INFO] Repo already exists at {path}, fetching updates...")
        out, err, code = run_cmd(["git", "fetch", "origin"], cwd=path)
        if code != 0:
            print(f"[ERROR] Error fetching in {path}:\n{err}")
            return False
        return True


def get_pr_commits_in_repo(path: Path, pr_number: int):
    """
    Assumes repo is already cloned/fetched at 'path'.
    Fetch PR as pr-<number>. Then try:

      git log --oneline main..pr-<number>
      if that fails, try:
      git log --oneline master..pr-<number>

    Returns:
      - list[str] on success (possibly empty if no unique commits)
      - None if both branches fail (not accessible)
    """
    pr_branch = f"pr-{pr_number}"

    # Fetch PR reference into local branch
    fetch_ref = f"pull/{pr_number}/head:{pr_branch}"
    print(f"[INFO] Fetching PR #{pr_number} as branch {pr_branch}")
    out, err, code = run_cmd(["git", "fetch", "origin", fetch_ref], cwd=path)
    if code != 0:
        print(f"[ERROR] Error fetching PR #{pr_number} in {path}:\n{err}")
        return None

    # Try main, then master
    for base_branch in ("main", "master"):
        range_spec = f"{base_branch}..{pr_branch}"
        print(f"[INFO] Trying: git log --oneline {range_spec}")
        out, err, code = run_cmd(["git", "log", "--oneline", range_spec], cwd=path)

        if code != 0:
            print(f"[WARN] git log failed for base '{base_branch}': {err}")
            continue  # try next base branch

        # Command succeeded
        if not out.strip():
            # No unique commits vs base branch
            return []

        commits = out.split("\n")
        return commits

    # If we reach here, both main and master failed
    print(f"[ERROR] Could not run git log for PR #{pr_number} with main/master.")
    return None


def process_dataframe_grouped(df: pd.DataFrame, results_csv: Path = RESULTS_CSV) -> pd.DataFrame:
    """
    df must have columns: 'html_url', 'number'

    We:
      - derive 'repo_url' from 'html_url'
      - group by 'repo_url'
      - for each repo, clone once, process all its PRs, then delete the repo
      - save results to CSV in "real time" (after each repo)

    Returns a DataFrame with one row per commit (or per PR if not accessible).

    Columns:
      - repo_url
      - pr_number
      - short_sha
      - message
      - status  ('ok', 'no commits', 'not accessible')
    """
    df = df.copy()

    # Derive repo_url once from html_url (PR URL)
    df["repo_url"] = df["html_url"].apply(lambda u: u.split("/pull", 1)[0] + ".git")

    # Remove old results file if it exists (fresh run)
    if results_csv.exists():
        results_csv.unlink()

    all_rows = []

    # Group all PRs by repo_url
    for repo_url, group in df.groupby("repo_url"):
        print("\n" + "#" * 80)
        print(f"[INFO] Processing repository: {repo_url}")
        print("#" * 80)

        path = repo_local_path(repo_url)
        repo_rows = []

        # Clone or fetch this repo ONCE
        ok = clone_or_update_repo(repo_url, path)
        if not ok:
            # Mark all PRs for this repo as not accessible
            print(f"[WARN] Repo {repo_url} not accessible, skipping all its PRs.")
            for _, row in group.iterrows():
                pr_number = int(row["number"])
                row_dict = {
                    "repo_url": repo_url,
                    "pr_number": pr_number,
                    "short_sha": None,
                    "message": None,
                    "status": "not accessible",
                }
                repo_rows.append(row_dict)
                all_rows.append(row_dict)
        else:
            # Repo is available: process each PR in this repo
            try:
                for _, row in group.iterrows():
                    pr_number = int(row["number"])
                    print("\n" + "=" * 80)
                    print(f"[INFO]   PR #{pr_number}")
                    print("=" * 80)

                    commits = get_pr_commits_in_repo(path, pr_number)

                    # PR not accessible (fetch/log failed for both branches)
                    if commits is None:
                        row_dict = {
                            "repo_url": repo_url,
                            "pr_number": pr_number,
                            "short_sha": None,
                            "message": None,
                            "status": "not accessible",
                        }
                        repo_rows.append(row_dict)
                        all_rows.append(row_dict)
                        continue

                    # No unique commits vs base branch
                    if not commits:
                        row_dict = {
                            "repo_url": repo_url,
                            "pr_number": pr_number,
                            "short_sha": None,
                            "message": None,
                            "status": "no commits",
                        }
                        repo_rows.append(row_dict)
                        all_rows.append(row_dict)
                        continue

                    # One row per commit
                    for line in commits:
                        parts = line.split(" ", 1)
                        short_sha = parts[0]
                        message = parts[1] if len(parts) > 1 else ""
                        row_dict = {
                            "repo_url": repo_url,
                            "pr_number": pr_number,
                            "short_sha": short_sha,
                            "message": message,
                            "status": "ok",
                        }
                        repo_rows.append(row_dict)
                        all_rows.append(row_dict)

            finally:
                # After processing all PRs for this repo, delete local clone
                if path.exists():
                    print(f"\n[INFO] Deleting local repo folder: {path}")
                    shutil.rmtree(path, ignore_errors=True)

        # 🔴 REAL-TIME SAVE: append this repo's rows to CSV
        if repo_rows:
            repo_df = pd.DataFrame(repo_rows)
            mode = "a"
            header = not results_csv.exists()
            repo_df.to_csv(results_csv, mode=mode, header=header, index=False)
            print(f"[INFO] Appended {len(repo_rows)} rows to {results_csv}")

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    # all_pr_df must exist and have columns: 'html_url', 'number'
    # Example:
    # all_pr_df = pd.read_csv("all_prs.csv")

    results_df = process_dataframe_grouped(all_pr_df, RESULTS_CSV)

    print("\n[INFO] Final results DataFrame (in memory):")
    print(results_df.head())
    print(f"[INFO] Full results saved incrementally to: {RESULTS_CSV}")
