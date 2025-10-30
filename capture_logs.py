#!/usr/bin/env python3
"""
Log Capture Script for SeamlessStreaming Server

This script captures detailed logs from the streaming server during testing.
"""

import logging
import sys
from datetime import datetime

# Create a custom log formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Add timestamp to each log record
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"[{timestamp}] {record.levelname}: {record.getMessage()}"

def setup_logging(log_file="streaming_server_logs.txt"):
    """Setup logging to capture all output to a file"""
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler  
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = CustomFormatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    print(f"✅ Logging setup complete - logs will be saved to: {log_file}")
    print("🚀 Starting server with detailed logging...")
    
    return log_file

if __name__ == "__main__":
    log_file = setup_logging()
    
    # Import and run the streaming server with logging enabled
    try:
        # Set environment variable to enable detailed logging
        import os
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Import the server module
        import streaming_server
        
        print(f"📁 Logs are being written to: {log_file}")
        print("💡 Run your test now, then check the log file")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Server stopped. Check logs in: {log_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"📁 Check logs in: {log_file}")