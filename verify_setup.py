"""
Setup Verification Script
Checks if the project is properly configured and ready to run
"""

import sys
import os
import importlib
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def check_python_version():
    """Check Python version"""
    print_header("1. Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python version: {version_str} (OK)")
        return True
    else:
        print_error(f"Python version: {version_str} (Need 3.8+)")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print_header("2. Checking Dependencies")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'requests': 'Requests',
        'textblob': 'TextBlob',
        'plotly': 'Plotly',
        'yfinance': 'yfinance',
        'joblib': 'Joblib',
        'nltk': 'NLTK',
        'vaderSentiment': 'VADER Sentiment'
    }
    
    optional_packages = {
        'xgboost': 'XGBoost',
        'tensorflow': 'TensorFlow'
    }
    
    all_good = True
    
    # Check required packages
    print("Required Packages:")
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            all_good = False
    
    # Check optional packages
    print("\nOptional Packages:")
    for package, name in optional_packages.items():
        try:
            importlib.import_module(package)
            print_success(f"{name} installed")
        except ImportError:
            print_warning(f"{name} NOT installed (optional)")
    
    return all_good


def check_nltk_data():
    """Check if NLTK data is downloaded"""
    print_header("3. Checking NLTK Data")
    
    try:
        nltk = importlib.import_module('nltk')
        
        required_data = {
            'tokenizers/punkt': 'Punkt tokenizer',
            'corpora/stopwords': 'Stopwords corpus'
        }
        
        all_good = True
        for data_id, name in required_data.items():
            try:
                nltk.data.find(data_id)
                print_success(f"{name} available")
            except LookupError:
                print_error(f"{name} NOT available")
                all_good = False
        
        if not all_good:
            print(f"\n{Colors.YELLOW}To download NLTK data, run:{Colors.RESET}")
            print('python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'stopwords\')"')
        
        return all_good
        
    except ImportError:
        print_error("NLTK not installed")
        return False


def check_project_files():
    """Check if all project files exist"""
    print_header("4. Checking Project Files")
    
    required_files = {
        'requirements.txt': 'Dependencies file',
        'data_collector.py': 'Data collection module',
        'sentiment_analyzer.py': 'Sentiment analysis module',
        'ml_models.py': 'ML models module',
        'train_models.py': 'Training script',
        'app.py': 'Streamlit application',
        'config.py': 'Configuration file'
    }
    
    optional_files = {
        'demo.py': 'Demo script',
        'README.md': 'Documentation',
        'SETUP_AND_RUN.md': 'Setup guide',
        'QUICKSTART.md': 'Quick start guide'
    }
    
    all_good = True
    
    print("Required Files:")
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print_success(f"{filename} - {description}")
        else:
            print_error(f"{filename} - {description} MISSING")
            all_good = False
    
    print("\nOptional Files:")
    for filename, description in optional_files.items():
        if os.path.exists(filename):
            print_success(f"{filename} - {description}")
        else:
            print_warning(f"{filename} - {description} (optional)")
    
    return all_good


def check_directories():
    """Check if required directories exist"""
    print_header("5. Checking Directories")
    
    required_dirs = ['models', 'logs']
    optional_dirs = ['data', 'reports', 'cache']
    
    print("Required Directories:")
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print_success(f"{dir_name}/ exists")
        else:
            print_warning(f"{dir_name}/ will be created automatically")
    
    print("\nOptional Directories:")
    for dir_name in optional_dirs:
        if os.path.exists(dir_name):
            print_success(f"{dir_name}/ exists")
        else:
            print_warning(f"{dir_name}/ (optional)")
    
    return True


def check_data_access():
    """Check if data can be fetched from APIs"""
    print_header("6. Checking Data Access")
    
    # Dynamically import yfinance to avoid static import errors if not installed
    try:
        yf = importlib.import_module('yfinance')
    except ImportError:
        print_error("yfinance NOT installed")
        print_warning("Install with: pip install yfinance")
        return False

    try:
        print("Testing yfinance API...")
        
        # Try to fetch a small amount of data
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period="1d")
        
        if not data.empty:
            print_success("yfinance API working (fetched BTC-USD data)")
            return True
        else:
            print_error("yfinance API not returning data")
            return False
            
    except Exception as e:
        print_error(f"Failed to fetch data: {str(e)}")
        print_warning("Check your internet connection")
        return False


def check_module_imports():
    """Check if project modules can be imported"""
    print_header("7. Checking Module Imports")
    
    modules = {
        'data_collector': 'Data Collector',
        'sentiment_analyzer': 'Sentiment Analyzer',
        'ml_models': 'ML Models',
        'config': 'Configuration'
    }
    
    all_good = True
    for module_name, description in modules.items():
        try:
            importlib.import_module(module_name)
            print_success(f"{description} can be imported")
        except Exception as e:
            print_error(f"{description} import failed: {str(e)}")
            all_good = False
    
    return all_good


def check_trained_models():
    """Check if any trained models exist"""
    print_header("8. Checking Trained Models")
    
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print_warning("No models directory found")
        print("  Run: python train_models.py [SYMBOL] [INTERVAL] [PERIOD]")
        return True  # Not a critical error
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if model_files:
        print_success(f"Found {len(model_files)} trained model(s):")
        for model_file in model_files[:5]:  # Show first 5
            print(f"  • {model_file}")
        if len(model_files) > 5:
            print(f"  ... and {len(model_files) - 5} more")
    else:
        print_warning("No trained models found")
        print("  To train a model, run:")
        print("    python train_models.py BTC-USD 1h 3mo")
    
    return True


def run_quick_test():
    """Run a quick functional test"""
    print_header("9. Running Quick Test")
    
    try:
        print("Testing data collection...")
        from data_collector import StockDataCollector
        
        collector = StockDataCollector('BTC-USD', '1h')
        df = collector.fetch_realtime_data('1d')
        
        if df is not None and not df.empty:
            print_success(f"Data collection works (fetched {len(df)} records)")
        else:
            print_error("Data collection failed")
            return False
        
        print("\nTesting sentiment analysis...")
        from sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_comprehensive("Bitcoin is showing strong growth!")
        
        if result and 'sentiment' in result:
            print_success(f"Sentiment analysis works (detected: {result['sentiment']})")
        else:
            print_error("Sentiment analysis failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def print_summary(results):
    """Print verification summary"""
    print_header("VERIFICATION SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r)
    failed_checks = total_checks - passed_checks
    
    print(f"Total Checks: {total_checks}")
    print(f"{Colors.GREEN}Passed: {passed_checks}{Colors.RESET}")
    if failed_checks > 0:
        print(f"{Colors.RED}Failed: {failed_checks}{Colors.RESET}")
    
    print("\nCheck Results:")
    for check_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {check_name}: {status}")
    
    print("\n" + "="*80)
    
    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED!{Colors.RESET}")
        print(f"\n{Colors.GREEN}Your system is ready to run the application!{Colors.RESET}")
        print("\nNext Steps:")
        print("  1. Train a model: python train_models.py BTC-USD 1h 3mo")
        print("  2. Run the app:  streamlit run app.py")
        print("  3. Open browser: http://localhost:8501")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠ SOME CHECKS FAILED{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Please fix the issues above before running the application.{Colors.RESET}")
        print("\nCommon Solutions:")
        print("  • Missing packages: pip install -r requirements.txt")
        print("  • NLTK data: python -c \"import nltk; nltk.download('all')\"")
        print("  • Check internet connection for data access")
    
    print("\n" + "="*80 + "\n")
    
    return all(results.values())


def main():
    """Main verification function"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{'Stock Price & Sentiment Predictor - Setup Verification'.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"\nVerification started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run all checks
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'NLTK Data': check_nltk_data(),
        'Project Files': check_project_files(),
        'Directories': check_directories(),
        'Data Access': check_data_access(),
        'Module Imports': check_module_imports(),
        'Trained Models': check_trained_models(),
        'Quick Test': run_quick_test()
    }
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
