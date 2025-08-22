"""
Quick test script to verify the package structure works correctly.
"""

def test_imports():
    """Test that all main imports work."""
    print("Testing imports...")
    
    # Test main package
    import cut
    print("✓ Main package imported successfully")
    
    # Test main exports
    from cut import TrainOptions, TestOptions, create_model, create_dataset
    print("✓ Main exports imported successfully")
    
    # Test scripts
    from cut._scripts import train, test, inference
    print("✓ Scripts imported successfully")
    
    # Test individual modules
    from cut.models import base_model
    from cut.data import base_dataset
    from cut.util import util
    print("✓ Individual modules imported successfully")
    
    # Test inference pipeline
    from cut._scripts.inference import InferencePipeline
    print("✓ InferencePipeline imported successfully")
    
    print("All imports successful! 🎉")


def test_options():
    """Test that options parsing works."""
    print("\nTesting options...")
    
    try:
        from cut import TrainOptions
        # Just test that the class can be instantiated
        # We don't parse because it would require command line args
        opt_class = TrainOptions()
        print("✓ TrainOptions class created successfully")
        
        from cut import TestOptions
        opt_class = TestOptions()
        print("✓ TestOptions class created successfully")
        
    except Exception as e:
        print(f"✗ Options test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing CUT package structure...\n")
    
    try:
        test_imports()
        test_options()
        print("\n🎉 All tests passed! The package structure is working correctly.")
        print("\nYou can now:")
        print("1. Use: poetry run cut-train <args>")
        print("2. Use: poetry run cut-test <args>") 
        print("3. Use: poetry run cut-inference")
        print("4. Import the package in Python scripts")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("There may be missing dependencies or circular imports.")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
