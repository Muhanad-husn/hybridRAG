import os
import shutil
import argparse
import logging
from datetime import datetime

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_remove(path):
    """Safely remove a file or directory."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logging.info(f"Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            logging.info(f"Removed directory: {path}")
    except Exception as e:
        logging.error(f"Error removing {path}: {str(e)}")

def cleanup_temp_files():
    """Clean up temporary files."""
    temp_dirs = [
        "data",
        "results",
        os.path.join("static", "assets", "temp"),
    ]
    for dir in temp_dirs:
        if os.path.exists(dir):
            safe_remove(dir)
    logging.info("Temporary files cleanup completed")

def cleanup_logs():
    """Clean up log files older than 7 days."""
    log_dir = "logs"
    if os.path.exists(log_dir):
        current_time = datetime.now()
        for file in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if (current_time - file_time).days > 7:
                    safe_remove(file_path)
    logging.info("Log files cleanup completed")

def cleanup_cache():
    """Clean up cache files."""
    cache_dirs = [
        "__pycache__",
        ".pytest_cache",
    ]
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if dir in cache_dirs:
                safe_remove(os.path.join(root, dir))
    logging.info("Cache cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="Cleanup script for the application environment")
    parser.add_argument("--all", action="store_true", help="Perform all cleanup operations")
    parser.add_argument("--temp", action="store_true", help="Clean up temporary files")
    parser.add_argument("--logs", action="store_true", help="Clean up old log files")
    parser.add_argument("--cache", action="store_true", help="Clean up cache files")
    args = parser.parse_args()

    if args.all or args.temp:
        cleanup_temp_files()
    if args.all or args.logs:
        cleanup_logs()
    if args.all or args.cache:
        cleanup_cache()

    if not (args.all or args.temp or args.logs or args.cache):
        parser.print_help()

    logging.info("Cleanup process completed")

if __name__ == "__main__":
    main()